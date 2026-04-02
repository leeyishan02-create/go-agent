// client.go 定义了与 OpenAI 兼容 API 进行通信的 HTTP 客户端。
// 该文件包含了所有与 LLM API 交互相关的数据结构和方法，
// 支持普通请求和 SSE（Server-Sent Events）流式请求两种模式。
// 客户端兼容 OpenAI、Ollama、DeepSeek 等多种 LLM API 服务。
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

// Message 表示一条对话消息。
// 该结构体用于构建发送给 LLM 的消息列表以及解析返回结果。
type Message struct {
	// Role 消息角色（string 类型），取值为 "system"、"user" 或 "assistant"
	Role string `json:"role"`
	// Content 消息的文本内容（string 类型）
	Content string `json:"content"`
}

// ToolDefinition 定义了一个可供模型调用的工具（函数）。
// 该结构体遵循 OpenAI function calling API 的格式。
type ToolDefinition struct {
	// Type 工具类型（string 类型），固定为 "function"
	Type string `json:"type"`
	// Function 工具函数的详细定义（匿名结构体），包含名称、描述和参数 Schema
	Function struct {
		// Name 函数名称（string 类型），如 "read"、"write"、"search"
		Name string `json:"name"`
		// Description 函数功能描述（string 类型），帮助模型理解何时使用该工具
		Description string `json:"description"`
		// Parameters 函数参数的 JSON Schema 定义（map[string]interface{} 类型）
		Parameters map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

// ToolCall 表示模型请求调用的一个工具。
// 当模型决定使用工具时，会在响应中包含 ToolCall 信息。
type ToolCall struct {
	// ID 工具调用的唯一标识符（string 类型），用于关联调用和结果
	ID string `json:"id"`
	// Type 调用类型（string 类型），固定为 "function"
	Type string `json:"type"`
	// Index 工具调用在响应中的索引（int 类型），用于流式响应中组装分片
	Index int `json:"index"`
	// Function 被调用函数的详细信息（匿名结构体）
	Function struct {
		// Name 函数名称（string 类型）
		Name string `json:"name"`
		// Arguments 函数参数的 JSON 字符串（string 类型），需要额外解析
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Choice 表示 API 响应中的一个选择项。
// 对于非流式请求，直接使用 Message 和 ToolCalls 字段；
// 对于流式请求，使用 Delta 字段来增量构建完整响应。
type Choice struct {
	// Index 选择项的索引（int 类型）
	Index int `json:"index"`
	// Message 完整的响应消息（Message 类型），用于非流式响应
	Message Message `json:"message"`
	// ToolCalls 模型请求调用的工具列表（[]ToolCall 类型），用于非流式响应
	ToolCalls []ToolCall `json:"tool_calls"`
	// FinishReason 响应完成原因（string 类型），如 "stop"、"tool_calls"
	FinishReason string `json:"finish_reason"`
	// Delta 流式响应的增量数据（DeltaChoice 类型），每个 chunk 包含部分内容
	Delta DeltaChoice `json:"delta"`
}

// DeltaChoice 表示流式响应的增量数据块。
// 在 SSE 流中，每个 chunk 包含一个 DeltaChoice，内容需要累加拼接。
type DeltaChoice struct {
	// Role 消息角色（string 类型），通常只在第一个 chunk 中出现
	Role string `json:"role"`
	// Content 增量文本内容（string 类型），需要与之前的 chunk 拼接
	Content string `json:"content"`
	// Reasoning 推理过程内容（string 类型），部分模型（如 DeepSeek）支持
	// omitempty 表示为空时不序列化到 JSON
	Reasoning string `json:"reasoning,omitempty"`
	// ToolCalls 增量工具调用信息（[]ToolCall 类型），参数可能分多个 chunk 发送
	ToolCalls []ToolCall `json:"tool_calls"`
	// FinishReason 响应完成原因（string 类型）
	FinishReason string `json:"finish_reason"`
}

// Usage 表示 API 调用的 token 使用量统计。
type Usage struct {
	// PromptTokens 提示词消耗的 token 数量（int 类型）
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens 模型回复消耗的 token 数量（int 类型）
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens 总 token 使用量（int 类型），等于 PromptTokens + CompletionTokens
	TotalTokens int `json:"total_tokens"`
}

// ChatResponse 表示聊天 API 的完整响应体。
type ChatResponse struct {
	// Choices 模型生成的选择项列表（[]Choice 类型），通常只包含一个元素
	Choices []Choice `json:"choices"`
	// Usage token 使用量统计信息（Usage 类型）
	Usage Usage `json:"usage"`
}

// ChatRequest 表示发送给聊天 API 的请求体。
type ChatRequest struct {
	// Model 模型名称（string 类型），如 "qwen2.5-coder"、"gpt-4" 等
	Model string `json:"model"`
	// Messages 对话消息列表（[]Message 类型），包含系统提示、历史消息和用户输入
	Messages []Message `json:"messages"`
	// Tools 可用工具定义列表（[]ToolDefinition 类型），为空时不包含在请求中
	// omitempty 表示为空时不序列化到 JSON
	Tools []ToolDefinition `json:"tools,omitempty"`
	// Stream 是否启用 SSE 流式传输（bool 类型）
	// true 表示服务器逐步返回生成的内容，false 表示等待完整生成后一次性返回
	Stream bool `json:"stream"`
	// MaxTokens 最大生成 token 数（int 类型），为 0 时不限制
	// omitempty 表示为 0 时不包含在请求中，让服务器使用默认值
	MaxTokens int `json:"max_tokens,omitempty"`
}

// defaultHTTPClient 是全局默认的 HTTP 客户端，配置了连接池和超时参数。
// 所有 API 请求默认使用该客户端，以提高连接复用效率。
// 配置说明：
//   - MaxIdleConns: 100，最大空闲连接总数
//   - MaxIdleConnsPerHost: 10，每个主机最大空闲连接数
//   - IdleConnTimeout: 90 秒，空闲连接超时时间
//   - DialContext Timeout: 30 秒，TCP 连接建立超时
//   - DialContext KeepAlive: 30 秒，TCP 连接保活间隔
//   - TLSHandshakeTimeout: 10 秒，TLS 握手超时
//   - ResponseHeaderTimeout: 120 秒，等待响应头超时
//   - ExpectContinueTimeout: 1 秒，100-continue 超时
//   - 总超时: 120 秒
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

// Client 封装了与 LLM API 交互的所有配置和方法。
// 通过 NewClient() 创建实例，或在 Web UI 中动态配置。
type Client struct {
	// BaseURL API 服务的基础 URL（string 类型），如 "http://localhost:11434/v1"
	BaseURL string
	// APIKey API 密钥（string 类型），用于身份验证
	APIKey string
	// Model 模型名称（string 类型），如 "qwen2.5-coder"、"gpt-4" 等
	Model string
	// HTTPClient 自定义的 HTTP 客户端（*http.Client 类型），为 nil 时使用 defaultHTTPClient
	HTTPClient *http.Client
}

// NewClient 创建一个新的 Client 实例。
// 配置项从环境变量中读取：
//   - OPENAI_BASE_URL：API 基础 URL，默认 "http://localhost:11434/v1"（Ollama 本地服务）
//   - OPENAI_API_KEY：API 密钥
//   - OPENAI_MODEL：模型名称，默认 "qwen2.5-coder"
//
// 返回值：*Client 类型的客户端实例指针
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

// httpClient 返回当前客户端使用的 HTTP 客户端。
// 如果客户端的 HTTPClient 字段已设置，返回该自定义 HTTP 客户端；
// 否则返回全局默认的 defaultHTTPClient。
// 返回值：*http.Client 类型的 HTTP 客户端指针
func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return defaultHTTPClient
}

// doRequest 发送一个聊天 API 请求并返回原始 HTTP 响应。
// 参数：
//   - ctx (context.Context)：请求上下文，用于超时和取消控制
//   - req (ChatRequest)：聊天请求体
//
// 返回值：
//   - *http.Response：HTTP 响应对象，调用者需要负责关闭响应体
//   - error：请求构建或发送失败时返回错误
//
// 该方法执行以下步骤：
//  1. 将 ChatRequest 序列化为 JSON
//  2. 构建 HTTP POST 请求（URL 为 BaseURL + "/chat/completions"）
//  3. 设置 Content-Type 和 Authorization 请求头
//  4. 发送请求并返回响应
func (c *Client) doRequest(ctx context.Context, req ChatRequest) (*http.Response, error) {
	// 将请求体序列化为 JSON 字节数组
	// body ([]byte)：序列化后的 JSON 数据
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 创建带上下文的 HTTP POST 请求
	// httpReq (*http.Request)：完整的 HTTP 请求对象
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// 设置请求头
	httpReq.Header.Set("Content-Type", "application/json")
	// 如果配置了 API 密钥，添加 Bearer Token 认证头
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	// 使用 HTTP 客户端发送请求
	return c.httpClient().Do(httpReq)
}

// doRequestWithRetry 带重试机制的请求发送方法。
// 在遇到网络错误或服务器 5xx 错误时，会自动重试指定次数。
// 参数：
//   - ctx (context.Context)：请求上下文
//   - req (ChatRequest)：聊天请求体
//   - maxRetries (int)：最大重试次数
//
// 返回值：
//   - *http.Response：成功时返回 HTTP 响应
//   - error：所有重试均失败时返回最后一次的错误
//
// 重试策略：
//   - 重试间隔随尝试次数线性递增（attempt * 1 秒）
//   - 遇到 context.Canceled 或 context.DeadlineExceeded 时立即终止，不重试
//   - 服务器 5xx 错误时重试，最后一次尝试即使是 5xx 也会返回响应
func (c *Client) doRequestWithRetry(ctx context.Context, req ChatRequest, maxRetries int) (*http.Response, error) {
	// lastErr 记录最后一次失败的错误信息
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		// 非首次尝试时，等待一段时间再重试（退避策略）
		if attempt > 0 {
			select {
			case <-ctx.Done():
				// 上下文已取消，立即返回
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
				// 等待 attempt 秒后继续重试
			}
		}

		// 发送请求
		resp, err := c.doRequest(ctx, req)
		if err != nil {
			// 如果是上下文取消或超时，立即返回，不重试
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			// 记录本次尝试的错误并继续重试
			lastErr = fmt.Errorf("attempt %d/%d: %w", attempt+1, maxRetries, err)
			continue
		}

		// 服务器错误（5xx）时，关闭响应体并重试
		// 但如果是最后一次尝试，则直接返回响应供调用者处理
		if resp.StatusCode >= 500 && attempt < maxRetries-1 {
			resp.Body.Close()
			lastErr = fmt.Errorf("attempt %d/%d: server error %d", attempt+1, maxRetries, resp.StatusCode)
			continue
		}

		// 请求成功或已到最后一次尝试
		return resp, nil
	}
	return nil, fmt.Errorf("all %d retries failed: %w", maxRetries, lastErr)
}

// Chat 发送非流式聊天请求并等待完整响应。
// 参数：
//   - ctx (context.Context)：请求上下文
//   - messages ([]Message)：对话消息列表
//   - tools ([]ToolDefinition)：可用工具定义列表
//
// 返回值：
//   - string：模型回复的文本内容
//   - []ToolCall：模型请求调用的工具列表（如果模型决定使用工具）
//   - Usage：token 使用量统计
//   - error：请求或解析失败时返回错误
//
// 该方法使用 doRequestWithRetry 发送请求，最多重试 2 次。
func (c *Client) Chat(ctx context.Context, messages []Message, tools []ToolDefinition) (string, []ToolCall, Usage, error) {
	// 构建请求体，Stream 设为 false 表示非流式请求
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	// 发送请求，最多重试 2 次
	resp, err := c.doRequestWithRetry(ctx, req, 2)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	// 读取完整响应体
	// respBody ([]byte)：API 返回的原始 JSON 数据
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("read response: %w", err)
	}

	// 检查 HTTP 状态码，非 200 视为错误
	if resp.StatusCode != 200 {
		return "", nil, Usage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	// 解析 JSON 响应为 ChatResponse 结构体
	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", nil, Usage{}, fmt.Errorf("parse response: %w", err)
	}

	// 验证响应中至少包含一个选择项
	if len(chatResp.Choices) == 0 {
		return "", nil, Usage{}, errors.New("no choices in response")
	}

	// 提取第一个选择项的内容和工具调用
	choice := chatResp.Choices[0]
	return choice.Message.Content, choice.ToolCalls, chatResp.Usage, nil
}

// ChatStream 发送流式聊天请求，实时处理 SSE（Server-Sent Events）数据流。
// 参数：
//   - ctx (context.Context)：请求上下文，可用于取消流式传输
//   - messages ([]Message)：对话消息列表
//   - tools ([]ToolDefinition)：可用工具定义列表
//   - onChunk (func(string))：接收文本内容增量的回调函数，每当收到新的文本 chunk 时被调用
//   - onReasoning (func(string))：接收推理过程增量的回调函数（部分模型支持显示推理过程）
//
// 返回值：
//   - string：完整的模型回复文本（所有 chunk 拼接后的结果）
//   - []ToolCall：模型请求调用的工具列表（从流中组装而成）
//   - Usage：token 使用量统计
//   - error：请求或流读取失败时返回错误
//
// SSE 数据流格式：
//   - 每行以 "data: " 开头，后跟 JSON 格式的 ChatResponse
//   - 流结束标记为 "data: [DONE]"
//   - 工具调用的参数可能分多个 chunk 发送，需要通过 Index 字段进行组装
func (c *Client) ChatStream(ctx context.Context, messages []Message, tools []ToolDefinition, onChunk func(string), onReasoning func(string)) (string, []ToolCall, Usage, error) {
	// 构建流式请求体
	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   true,
	}

	// 发送请求（流式请求不使用重试，因为需要实时处理数据）
	resp, err := c.doRequest(ctx, req)
	if err != nil {
		return "", nil, Usage{}, fmt.Errorf("stream request: %w", err)
	}
	defer resp.Body.Close()

	// 检查 HTTP 状态码
	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return "", nil, Usage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	// fullContent (string)：累积的完整回复文本
	var fullContent string
	// allToolCalls ([]ToolCall)：最终组装完成的工具调用列表
	var allToolCalls []ToolCall
	// totalUsage (Usage)：token 使用量统计
	var totalUsage Usage
	// pendingCalls (map[int]*ToolCall)：正在组装中的工具调用，以 Index 为键
	// 因为流式响应中工具调用的参数可能分多个 chunk 发送，需要逐步拼接
	pendingCalls := make(map[int]*ToolCall)

	// 逐行读取 SSE 数据流
	// scanner (*bufio.Scanner)：用于逐行读取响应体
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		// 检查上下文是否已取消
		select {
		case <-ctx.Done():
			return fullContent, allToolCalls, totalUsage, ctx.Err()
		default:
		}

		// line (string)：当前读取的一行数据
		line := scanner.Text()
		// 跳过不以 "data: " 开头的行（如空行、注释等）
		if !bytes.HasPrefix([]byte(line), []byte("data: ")) {
			continue
		}
		// data (string)：去掉 "data: " 前缀后的数据内容
		data := line[6:]
		// "[DONE]" 表示流传输结束
		if data == "[DONE]" {
			break
		}

		// 解析当前 chunk 的 JSON 数据
		var chunk ChatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // 跳过无法解析的 chunk
		}

		// 跳过没有选择项的 chunk
		if len(chunk.Choices) == 0 {
			continue
		}

		// 更新 token 使用量（部分 API 在流的最后一个 chunk 中包含 usage 信息）
		if chunk.Usage.TotalTokens > 0 {
			totalUsage = chunk.Usage
		}

		// delta (DeltaChoice)：当前 chunk 的增量数据
		delta := chunk.Choices[0].Delta

		// 处理工具调用的增量数据
		// 流式响应中，工具调用的名称、ID 和参数可能分散在不同的 chunk 中
		// 需要通过 Index 字段将属于同一个工具调用的 chunk 合并
		if delta.Role == "assistant" && len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				// idx (int)：工具调用的索引，用于标识是哪个工具调用
				idx := tc.Index
				if existing, ok := pendingCalls[idx]; ok {
					// 已有该索引的工具调用，拼接参数字符串
					existing.Function.Arguments += tc.Function.Arguments
					// 更新名称和 ID（如果本 chunk 中包含这些信息）
					if tc.Function.Name != "" {
						existing.Function.Name = tc.Function.Name
					}
					if tc.ID != "" {
						existing.ID = tc.ID
					}
				} else {
					// 新的工具调用，创建副本并存入 pendingCalls
					tcCopy := tc
					pendingCalls[idx] = &tcCopy
				}
			}
		}

		// 处理推理过程增量
		// 部分模型（如 DeepSeek）支持输出推理过程，通过 onReasoning 回调传递
		if delta.Reasoning != "" && onReasoning != nil {
			onReasoning(delta.Reasoning)
		}

		// 处理文本内容增量
		// 将新的文本片段追加到 fullContent，并通过 onChunk 回调实时通知调用者
		if delta.Content != "" {
			fullContent += delta.Content
			if onChunk != nil {
				onChunk(delta.Content)
			}
		}
	}

	// 将 pendingCalls 中的工具调用按索引顺序取出，组装为最终列表
	// 从索引 0 开始连续取，遇到不存在的索引时停止
	for i := 0; ; i++ {
		if tc, ok := pendingCalls[i]; ok {
			allToolCalls = append(allToolCalls, *tc)
		} else {
			break
		}
	}

	// 检查 scanner 是否有读取错误
	if err := scanner.Err(); err != nil {
		return fullContent, allToolCalls, totalUsage, fmt.Errorf("stream read: %w", err)
	}
	return fullContent, allToolCalls, totalUsage, nil
}
