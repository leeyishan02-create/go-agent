package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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

type Client struct {
	BaseURL string
	APIKey  string
	Model   string
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
		BaseURL: baseURL,
		APIKey:  apiKey,
		Model:   model,
	}
}

func (c *Client) doRequest(req ChatRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequest("POST", c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	return http.DefaultClient.Do(httpReq)
}

func (c *Client) Chat(messages []Message, tools []ToolDefinition) (string, []ToolCall, Usage, error) {
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return "", nil, Usage{}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, Usage{}, err
	}

	if resp.StatusCode != 200 {
		return "", nil, Usage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", nil, Usage{}, err
	}

	if len(chatResp.Choices) == 0 {
		return "", nil, Usage{}, fmt.Errorf("no choices in response")
	}

	choice := chatResp.Choices[0]
	return choice.Message.Content, choice.ToolCalls, chatResp.Usage, nil
}

func (c *Client) ChatStream(messages []Message, tools []ToolDefinition, onChunk func(string), onReasoning func(string)) (string, []ToolCall, Usage, error) {
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   true,
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return "", nil, Usage{}, err
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

	return fullContent, allToolCalls, totalUsage, scanner.Err()
}
