// query.go 实现了 AI 代理的核心查询逻辑，包括系统提示词构建、事件系统、
// 工具调用循环以及多种查询接口。
// 该文件是连接 LLM 客户端和工具系统的核心桥梁。
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"go-agent/tools"
)

// baseSystemPrompt 是基础系统提示词，定义 AI 助手的角色。
const baseSystemPrompt = `You are a helpful AI assistant. You can use tools to accomplish tasks.`

// toolDescriptions 工具名称到描述文本的映射，用于在系统提示词中列出可用工具。
var toolDescriptions = map[string]string{
	"read":   "- read: Read file contents",
	"write":  "- write: Create, overwrite, or edit files",
	"search": "- search: Search the web for current information",
}

// rulesText 定义 AI 助手的行为规则，附加到系统提示词中。
const rulesText = `
Rules:
- Always think step by step before using tools
- Use the most appropriate tool for each task
- When editing files, use old_string/new_string for partial changes
- Report results clearly to the user
- If a command fails, try to understand why and fix it
- If a tool is unavailable, answer the question using your own knowledge instead
- If you cannot complete a task with available tools, explain what you can do and what you cannot`

// buildSystemPrompt 根据工具列表构建完整的系统提示词。
// 参数 toolList ([]Tool)：当前可用的工具列表。
// 返回值 string：包含基础提示、工具列表和规则的完整提示词。
func buildSystemPrompt(toolList []Tool) string {
	var sb strings.Builder
	sb.WriteString(baseSystemPrompt)
	sb.WriteString("\n\nAvailable tools:\n")
	for _, tool := range toolList {
		if desc, ok := toolDescriptions[tool.Name()]; ok {
			sb.WriteString(desc)
			sb.WriteString("\n")
		}
	}
	sb.WriteString(rulesText)
	return sb.String()
}

// EventType 事件类型枚举（string 类型），用于标识查询过程中不同阶段的事件。
type EventType string

const (
	EventThinking   EventType = "thinking"    // 模型思考/推理中
	EventAnswer     EventType = "answer"      // 模型生成的回答内容
	EventToolUse    EventType = "tool_use"    // 模型请求调用工具
	EventToolResult EventType = "tool_result" // 工具执行结果
	EventDone       EventType = "done"        // 查询完成
	EventError      EventType = "error"       // 发生错误
)

// Event 表示查询过程中产生的事件，通过 EventCallback 传递给调用者。
// Web UI 模式下会序列化为 JSON 通过 SSE 发送给前端。
type Event struct {
	Type        EventType       `json:"type"`                   // 事件类型
	Content     string          `json:"content,omitempty"`      // 事件内容（思考/回答/错误文本）
	Tool        string          `json:"tool,omitempty"`         // 工具名称
	ID          string          `json:"id,omitempty"`           // 工具调用 ID
	Input       json.RawMessage `json:"input,omitempty"`        // 工具输入参数（原始 JSON）
	Output      string          `json:"output,omitempty"`       // 工具执行结果
	AgentID     string          `json:"agent_id,omitempty"`     // 多代理模式中的代理标识
	AgentStatus string          `json:"agent_status,omitempty"` // 代理状态（thinking/debating/done）
	Usage       UsageInfo       `json:"usage,omitempty"`        // token 使用量（仅 EventDone）
}

// UsageInfo 表示 token 使用量统计，用于事件系统中传递数据。
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`     // 提示词 token 数
	CompletionTokens int `json:"completion_tokens"` // 回复 token 数
	TotalTokens      int `json:"total_tokens"`      // 总 token 数
}

// EventCallback 事件回调函数类型，每产生一个事件就调用一次。
type EventCallback func(event Event)

// buildToolDefs 将 Tool 接口列表转换为 OpenAI function calling 格式的 ToolDefinition 列表。
// 参数 toolList ([]Tool)：工具接口列表。
// 返回值 []ToolDefinition：API 所需的工具定义列表。
func buildToolDefs(toolList []Tool) []ToolDefinition {
	defs := make([]ToolDefinition, len(toolList))
	for i, tool := range toolList {
		defs[i] = ToolDefinition{Type: "function"}
		defs[i].Function.Name = tool.Name()
		defs[i].Function.Description = tool.Description()
		defs[i].Function.Parameters = tool.InputSchema()
	}
	return defs
}

// findTool 根据名称在工具列表中查找工具，未找到返回 nil。
// 参数 toolList ([]Tool)：工具列表；name (string)：工具名称。
func findTool(toolList []Tool, name string) Tool {
	for _, t := range toolList {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

// Query 命令行交互模式的简单查询接口，将结果直接打印到标准输出。
// 参数：ctx 上下文, client API 客户端, toolList 工具列表, userMessage 用户消息。
func Query(ctx context.Context, client *Client, toolList []Tool, userMessage string) error {
	_, err := QueryWithCallbackAndCtx(ctx, client, toolList, nil, userMessage, func(event Event) {
		switch event.Type {
		case EventThinking:
			fmt.Print(event.Content)
		case EventAnswer:
			fmt.Print(event.Content)
		case EventToolUse:
			fmt.Printf("\n🔧 %s(%s)\n", event.Tool, truncate(string(event.Input), 80))
		case EventToolResult:
			fmt.Printf("📄 %s\n", truncate(event.Output, 200))
		case EventError:
			fmt.Fprintf(os.Stderr, "Error: %s\n", event.Content)
		}
	})
	return err
}

// QueryWithCallback 带自定义事件回调的查询接口（无历史消息）。
func QueryWithCallback(client *Client, toolList []Tool, userMessage string, cb EventCallback) error {
	_, err := QueryWithCallbackAndHistory(client, toolList, nil, userMessage, cb)
	return err
}

// QueryWithCallbackAndHistory 带回调和历史消息的查询接口，支持多轮对话。
// 参数 history ([]Message)：历史对话消息列表，可为 nil。
// 返回值：模型最终回复文本和错误。
func QueryWithCallbackAndHistory(client *Client, toolList []Tool, history []Message, userMessage string, cb EventCallback) (string, error) {
	return QueryWithCallbackAndCtx(context.Background(), client, toolList, history, userMessage, cb)
}

// QueryWithCallbackAndCtx 完整功能的查询函数，所有查询接口的底层实现。
// 包含完整的工具调用循环：发送请求→处理回复→执行工具→反馈结果→继续请求。
// 参数：ctx 上下文, client 客户端, toolList 工具, history 历史消息, userMessage 用户消息, cb 回调。
// 返回值：模型最终回复文本和错误。
// 最多执行 20 轮工具调用循环（maxTurns），防止无限循环。
func QueryWithCallbackAndCtx(ctx context.Context, client *Client, toolList []Tool, history []Message, userMessage string, cb EventCallback) (string, error) {
	systemPrompt := buildSystemPrompt(toolList)
	// 构建消息列表：系统提示 + 历史消息 + 当前用户消息
	var messages []Message
	messages = append(messages, Message{Role: "system", Content: systemPrompt})
	messages = append(messages, history...)
	messages = append(messages, Message{Role: "user", Content: userMessage})

	toolDefs := buildToolDefs(toolList)

	maxTurns := 20         // 最大工具调用轮次
	var totalUsage Usage   // 累计 token 使用量
	var assistantContent string // 模型最新回复

	// 工具调用循环
	for turn := 0; turn < maxTurns; turn++ {
		// 检查上下文是否已取消
		select {
		case <-ctx.Done():
			cb(Event{Type: EventError, Content: "query cancelled"})
			return assistantContent, ctx.Err()
		default:
		}

		// 发送流式请求，实时回调文本和推理内容
		content, toolCalls, usage, err := client.ChatStream(ctx, messages, toolDefs, func(chunk string) {
			cb(Event{Type: EventAnswer, Content: chunk})
		}, func(reasoning string) {
			cb(Event{Type: EventThinking, Content: reasoning})
		})
		if err != nil {
			cb(Event{Type: EventError, Content: err.Error()})
			return "", fmt.Errorf("chat error: %w", err)
		}

		// 累计 token 使用量
		if usage.TotalTokens > 0 {
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
		}

		// 记录模型回复并追加到消息列表
		if content != "" {
			assistantContent = content
			messages = append(messages, Message{Role: "assistant", Content: content})
		}

		// 无工具调用则查询完成
		if len(toolCalls) == 0 {
			cb(Event{Type: EventDone, Usage: UsageInfo{
				PromptTokens:     totalUsage.PromptTokens,
				CompletionTokens: totalUsage.CompletionTokens,
				TotalTokens:      totalUsage.TotalTokens,
			}})
			return assistantContent, nil
		}

		// 执行每个工具调用
		for _, tc := range toolCalls {
			tool := findTool(toolList, tc.Function.Name)
			if tool == nil {
				// 工具不存在，通知模型使用自身知识回答
				cb(Event{Type: EventError, Content: fmt.Sprintf("Tool %q is not available. Please answer using your own knowledge.", tc.Function.Name)})
				messages = append(messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("The tool %q is not available. Please answer the question using your own knowledge without calling any tools.", tc.Function.Name),
				})
				continue
			}

			cb(Event{Type: EventToolUse, Tool: tc.Function.Name, ID: tc.ID, Input: json.RawMessage(tc.Function.Arguments)})

			// 执行工具并获取结果
			result, err := tool.Execute(json.RawMessage(tc.Function.Arguments))
			if err != nil {
				result = "Error: " + err.Error()
			}

			cb(Event{Type: EventToolResult, ID: tc.ID, Tool: tc.Function.Name, Output: result})

			// 将结果追加到消息列表作为下一轮请求的上下文
			resultMsg := fmt.Sprintf("Tool call %s (%s) returned:\n%s", tc.ID, tc.Function.Name, result)
			messages = append(messages, Message{Role: "user", Content: resultMsg})
		}
	}

	// 超过最大轮次
	cb(Event{Type: EventError, Content: fmt.Sprintf("exceeded max turns (%d)", maxTurns)})
	return assistantContent, fmt.Errorf("exceeded max turns (%d)", maxTurns)
}

// truncate 将字符串截断到指定长度，替换换行符为空格，超出部分用 "..." 代替。
// 参数 s (string)：原始字符串；max (int)：最大字符数。
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

// GetTools 返回默认工具列表（ReadTool + WriteTool），不含搜索工具。
// 会自动设置工具的当前工作目录。
func GetTools() []Tool {
	tools.SetCwd(getCwd())
	return []Tool{
		&tools.ReadTool{},
		&tools.WriteTool{},
	}
}

// GetToolsWithSearch 返回包含搜索工具的工具列表。
// 参数 searchTool (Tool)：搜索工具实例，为 nil 时不添加搜索工具。
func GetToolsWithSearch(searchTool Tool) []Tool {
	tools.SetCwd(getCwd())
	t := []Tool{
		&tools.ReadTool{},
		&tools.WriteTool{},
	}
	if searchTool != nil {
		t = append(t, searchTool)
	}
	return t
}

// getCwd 获取当前工作目录路径，失败时返回 "."。
func getCwd() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return cwd
}
