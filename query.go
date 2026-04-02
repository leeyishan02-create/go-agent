package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"go-agent/tools"
)

const baseSystemPrompt = `You are a helpful AI assistant. You can use tools to accomplish tasks.`

var toolDescriptions = map[string]string{
	"read":   "- read: Read file contents",
	"write":  "- write: Create, overwrite, or edit files",
	"search": "- search: Search the web for current information",
}

const rulesText = `
Rules:
- Always think step by step before using tools
- Use the most appropriate tool for each task
- When editing files, use old_string/new_string for partial changes
- Report results clearly to the user
- If a command fails, try to understand why and fix it
- If a tool is unavailable, answer the question using your own knowledge instead
- If you cannot complete a task with available tools, explain what you can do and what you cannot`

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

type EventType string

const (
	EventThinking   EventType = "thinking"
	EventAnswer     EventType = "answer"
	EventToolUse    EventType = "tool_use"
	EventToolResult EventType = "tool_result"
	EventDone       EventType = "done"
	EventError      EventType = "error"
)

type Event struct {
	Type        EventType       `json:"type"`
	Content     string          `json:"content,omitempty"`
	Tool        string          `json:"tool,omitempty"`
	ID          string          `json:"id,omitempty"`
	Input       json.RawMessage `json:"input,omitempty"`
	Output      string          `json:"output,omitempty"`
	AgentID     string          `json:"agent_id,omitempty"`
	AgentStatus string          `json:"agent_status,omitempty"`
	Usage       UsageInfo       `json:"usage,omitempty"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type EventCallback func(event Event)

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

func findTool(toolList []Tool, name string) Tool {
	for _, t := range toolList {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

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

func QueryWithCallback(client *Client, toolList []Tool, userMessage string, cb EventCallback) error {
	_, err := QueryWithCallbackAndHistory(client, toolList, nil, userMessage, cb)
	return err
}

func QueryWithCallbackAndHistory(client *Client, toolList []Tool, history []Message, userMessage string, cb EventCallback) (string, error) {
	return QueryWithCallbackAndCtx(context.Background(), client, toolList, history, userMessage, cb)
}

func QueryWithCallbackAndCtx(ctx context.Context, client *Client, toolList []Tool, history []Message, userMessage string, cb EventCallback) (string, error) {
	systemPrompt := buildSystemPrompt(toolList)
	var messages []Message
	messages = append(messages, Message{Role: "system", Content: systemPrompt})
	messages = append(messages, history...)
	messages = append(messages, Message{Role: "user", Content: userMessage})

	toolDefs := buildToolDefs(toolList)

	maxTurns := 20
	var totalUsage Usage
	var assistantContent string
	for turn := 0; turn < maxTurns; turn++ {
		select {
		case <-ctx.Done():
			cb(Event{Type: EventError, Content: "query cancelled"})
			return assistantContent, ctx.Err()
		default:
		}

		content, toolCalls, usage, err := client.ChatStream(ctx, messages, toolDefs, func(chunk string) {
			cb(Event{Type: EventAnswer, Content: chunk})
		}, func(reasoning string) {
			cb(Event{Type: EventThinking, Content: reasoning})
		})
		if err != nil {
			cb(Event{Type: EventError, Content: err.Error()})
			return "", fmt.Errorf("chat error: %w", err)
		}

		if usage.TotalTokens > 0 {
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
		}

		if content != "" {
			assistantContent = content
			messages = append(messages, Message{Role: "assistant", Content: content})
		}

		if len(toolCalls) == 0 {
			cb(Event{Type: EventDone, Usage: UsageInfo{
				PromptTokens:     totalUsage.PromptTokens,
				CompletionTokens: totalUsage.CompletionTokens,
				TotalTokens:      totalUsage.TotalTokens,
			}})
			return assistantContent, nil
		}

		for _, tc := range toolCalls {
			tool := findTool(toolList, tc.Function.Name)
			if tool == nil {
				cb(Event{Type: EventError, Content: fmt.Sprintf("Tool %q is not available. Please answer using your own knowledge.", tc.Function.Name)})
				messages = append(messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("The tool %q is not available. Please answer the question using your own knowledge without calling any tools.", tc.Function.Name),
				})
				continue
			}

			cb(Event{Type: EventToolUse, Tool: tc.Function.Name, ID: tc.ID, Input: json.RawMessage(tc.Function.Arguments)})

			result, err := tool.Execute(json.RawMessage(tc.Function.Arguments))
			if err != nil {
				result = "Error: " + err.Error()
			}

			cb(Event{Type: EventToolResult, ID: tc.ID, Tool: tc.Function.Name, Output: result})

			resultMsg := fmt.Sprintf("Tool call %s (%s) returned:\n%s", tc.ID, tc.Function.Name, result)
			messages = append(messages, Message{Role: "user", Content: resultMsg})
		}
	}

	cb(Event{Type: EventError, Content: fmt.Sprintf("exceeded max turns (%d)", maxTurns)})
	return assistantContent, fmt.Errorf("exceeded max turns (%d)", maxTurns)
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func GetTools() []Tool {
	tools.SetCwd(getCwd())
	return []Tool{
		&tools.ReadTool{},
		&tools.WriteTool{},
	}
}

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

func getCwd() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return cwd
}
