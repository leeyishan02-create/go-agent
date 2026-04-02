package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ToolDefinition struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Index    int    `json:"index"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type Choice struct {
	Index        int         `json:"index"`
	Message      Message     `json:"message"`
	ToolCalls    []ToolCall  `json:"tool_calls"`
	FinishReason string      `json:"finish_reason"`
	Delta        DeltaChoice `json:"delta"`
}

type DeltaChoice struct {
	Role         string     `json:"role"`
	Content      string     `json:"content"`
	Reasoning    string     `json:"reasoning,omitempty"`
	ToolCalls    []ToolCall `json:"tool_calls"`
	FinishReason string     `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatResponse struct {
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type ChatRequest struct {
	Model     string           `json:"model"`
	Messages  []Message        `json:"messages"`
	Tools     []ToolDefinition `json:"tools,omitempty"`
	Stream    bool             `json:"stream"`
	MaxTokens int              `json:"max_tokens,omitempty"`
}

var defaultHTTPClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 120 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
	Timeout: 120 * time.Second,
}

type Client struct {
	BaseURL    string
	APIKey     string
	Model      string
	HTTPClient *http.Client
}

func NewClient() *Client {
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	model := os.Getenv("OPENAI_MODEL")
	if model == "" {
		model = "qwen2.5-coder"
	}
	return &Client{
		BaseURL:    baseURL,
		APIKey:     apiKey,
		Model:      model,
		HTTPClient: defaultHTTPClient,
	}
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return defaultHTTPClient
}

func (c *Client) doRequest(ctx context.Context, req ChatRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	return c.httpClient().Do(httpReq)
}

func (c *Client) doRequestWithRetry(ctx context.Context, req ChatRequest, maxRetries int) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}

		resp, err := c.doRequest(ctx, req)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			lastErr = fmt.Errorf("attempt %d/%d: %w", attempt+1, maxRetries, err)
			continue
		}

		if resp.StatusCode >= 500 && attempt < maxRetries-1 {
			resp.Body.Close()
			lastErr = fmt.Errorf("attempt %d/%d: server error %d", attempt+1, maxRetries, resp.StatusCode)
			continue
		}

		return resp, nil
	}
	return nil, fmt.Errorf("all %d retries failed: %w", maxRetries, lastErr)
}

func (c *Client) Chat(ctx context.Context, messages []Message, tools []ToolDefinition) (string, []ToolCall, Usage, error) {
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	resp, err := c.doRequestWithRetry(ctx, req, 2)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", nil, Usage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", nil, Usage{}, fmt.Errorf("parse response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", nil, Usage{}, errors.New("no choices in response")
	}

	choice := chatResp.Choices[0]
	return choice.Message.Content, choice.ToolCalls, chatResp.Usage, nil
}

func (c *Client) ChatStream(ctx context.Context, messages []Message, tools []ToolDefinition, onChunk func(string), onReasoning func(string)) (string, []ToolCall, Usage, error) {
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   true,
	}

	resp, err := c.doRequest(ctx, req)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("stream request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return "", nil, Usage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var fullContent string
	var allToolCalls []ToolCall
	var totalUsage Usage
	pendingCalls := make(map[int]*ToolCall)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return fullContent, allToolCalls, totalUsage, ctx.Err()
		default:
		}

		line := scanner.Text()
		if !bytes.HasPrefix([]byte(line), []byte("data: ")) {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			break
		}

		var chunk ChatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		if chunk.Usage.TotalTokens > 0 {
			totalUsage = chunk.Usage
		}

		delta := chunk.Choices[0].Delta

		if delta.Role == "assistant" && len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				idx := tc.Index
				if existing, ok := pendingCalls[idx]; ok {
					existing.Function.Arguments += tc.Function.Arguments
					if tc.Function.Name != "" {
						existing.Function.Name = tc.Function.Name
					}
					if tc.ID != "" {
						existing.ID = tc.ID
					}
				} else {
					tcCopy := tc
					pendingCalls[idx] = &tcCopy
				}
			}
		}

		if delta.Reasoning != "" && onReasoning != nil {
			onReasoning(delta.Reasoning)
		}

		if delta.Content != "" {
			fullContent += delta.Content
			if onChunk != nil {
				onChunk(delta.Content)
			}
		}
	}

	for i := 0; ; i++ {
		if tc, ok := pendingCalls[i]; ok {
			allToolCalls = append(allToolCalls, *tc)
		} else {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return fullContent, allToolCalls, totalUsage, fmt.Errorf("stream read: %w", err)
	}
	return fullContent, allToolCalls, totalUsage, nil
}
