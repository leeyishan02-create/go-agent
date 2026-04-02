// multiagent.go 实现了多代理查询系统，允许多个 AI 代理同时分析同一个问题，
// 然后通过"辩论"和"综合"阶段产生更全面、更高质量的回答。
//
// 多代理查询的三个阶段：
//  1. 初始回答：多个代理（Analyst、Researcher、Reviewer 等）并行生成各自的回答
//  2. 辩论轮次：每个代理查看其他代理的回答，改进自己的答案
//  3. 综合阶段：Synthesizer 代理汇总所有辩论后的回答，生成最终综合答案
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// AgentRole 定义了一个代理的角色信息。
type AgentRole struct {
	// Name 代理名称（string 类型），如 "Analyst"、"Researcher" 等
	Name string
	// Prefix 角色前缀提示词（string 类型），定义该代理的行为方式和分析角度
	Prefix string
}

// defaultRoles 默认的代理角色列表，每个角色有不同的分析视角。
var defaultRoles = []AgentRole{
	{Name: "Analyst", Prefix: "You are an analytical assistant. Break down the problem, examine it from multiple angles, and provide a thorough analysis."},
	{Name: "Researcher", Prefix: "You are a research-focused assistant. Gather relevant information, consider edge cases, and provide well-researched insights."},
	{Name: "Reviewer", Prefix: "You are a critical reviewer. Evaluate the problem carefully, identify potential issues, and suggest improvements."},
	{Name: "Synthesizer", Prefix: "You are a synthesis expert. Your job is to combine multiple perspectives into a single, coherent, and comprehensive answer."},
}

// AgentResult 存储单个代理的回答结果。
type AgentResult struct {
	Role    string // 代理角色名称
	Content string // 回答内容
	Error   error  // 执行过程中的错误（如果有）
}

// MultiAgentQuery 简单的多代理查询接口（无历史消息）。
// 参数：client 客户端, toolList 工具, userMessage 用户消息, agentCount 代理数量, cb 事件回调。
func MultiAgentQuery(client *Client, toolList []Tool, userMessage string, agentCount int, cb EventCallback) error {
	_, err := MultiAgentQueryWithHistory(client, toolList, nil, userMessage, agentCount, cb)
	return err
}

// MultiAgentQueryWithHistory 带历史消息的多代理查询接口。
// 自动创建可取消的上下文。
func MultiAgentQueryWithHistory(client *Client, toolList []Tool, history []Message, userMessage string, agentCount int, cb EventCallback) (string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	return multiAgentQueryWithCtx(ctx, client, toolList, history, userMessage, agentCount, cb)
}

// MultiAgentQueryWithCtx 带上下文的多代理查询接口（公开版本）。
func MultiAgentQueryWithCtx(ctx context.Context, client *Client, toolList []Tool, history []Message, userMessage string, agentCount int, cb EventCallback) (string, error) {
	return multiAgentQueryWithCtx(ctx, client, toolList, history, userMessage, agentCount, cb)
}

// multiAgentQueryWithCtx 多代理查询的核心实现。
//
// 执行流程：
//  1. 根据 agentCount 确定代理数量和角色（使用 defaultRoles 或自动生成）
//  2. 并行启动所有代理，每个代理独立调用 LLM 生成初始回答
//  3. 收集有效结果，过滤失败的代理
//  4. 如果只有一个有效结果，直接返回
//  5. 辩论阶段：每个代理查看其他代理的回答后改进自己的答案
//  6. 综合阶段：使用最后一个角色（Synthesizer）汇总所有辩论结果
//
// 参数：
//   - ctx：上下文, client：客户端, toolList：工具列表
//   - history：历史消息, userMessage：用户消息
//   - agentCount：代理数量（最少 1）
//   - cb：事件回调（用于实时通知进度）
func multiAgentQueryWithCtx(ctx context.Context, client *Client, toolList []Tool, history []Message, userMessage string, agentCount int, cb EventCallback) (string, error) {
	// 确保至少有 1 个代理
	if agentCount < 1 {
		agentCount = 1
	}

	// 确定使用的角色列表：优先使用预定义角色，超出时自动生成
	roles := defaultRoles[:min(agentCount, len(defaultRoles))]
	if agentCount > len(defaultRoles) {
		for i := len(defaultRoles); i < agentCount; i++ {
			roles = append(roles, AgentRole{
				Name:   fmt.Sprintf("Agent-%d", i+1),
				Prefix: fmt.Sprintf("You are assistant #%d. Provide a unique and thoughtful perspective on the problem.", i+1),
			})
		}
	}

	// agentWork 封装了每个代理任务的信息
	type agentWork struct {
		idx  int       // 代理在结果数组中的索引
		role AgentRole // 代理角色
	}

	var wg sync.WaitGroup
	results := make([]AgentResult, len(roles)) // 存储每个代理的结果
	var usageMu sync.Mutex                     // 保护 totalUsage 的并发写入
	var totalUsage Usage                       // 累计所有代理的 token 使用量

	// 构建工作任务列表
	works := make([]agentWork, len(roles))
	for i, role := range roles {
		works[i] = agentWork{idx: i, role: role}
	}

	// === 第一阶段：并行生成初始回答 ===
	for _, w := range works {
		wg.Add(1)
		go func(work agentWork) {
			defer wg.Done()

			// 为每个代理构建独立的消息列表（角色提示 + 历史 + 用户消息）
			var messages []Message
			messages = append(messages, Message{Role: "system", Content: work.role.Prefix + "\n\n" + buildSystemPrompt(toolList)})
			messages = append(messages, history...)
			messages = append(messages, Message{Role: "user", Content: userMessage})

			toolDefs := buildToolDefs(toolList)

			// 通知前端代理开始思考
			cb(Event{Type: EventThinking, Content: "starting...", AgentID: work.role.Name, AgentStatus: "thinking"})

			// 调用 LLM 生成回答
			content, _, usage, err := client.ChatStream(ctx, messages, toolDefs, nil, nil)
			if err != nil {
				cb(Event{Type: EventThinking, Content: "error: " + err.Error(), AgentID: work.role.Name, AgentStatus: "error"})
				results[work.idx] = AgentResult{Role: work.role.Name, Error: err}
				return
			}

			// 检查回答是否为空
			if strings.TrimSpace(content) == "" {
				cb(Event{Type: EventThinking, Content: "empty response", AgentID: work.role.Name, AgentStatus: "error"})
				results[work.idx] = AgentResult{Role: work.role.Name, Error: fmt.Errorf("empty response")}
				return
			}

			cb(Event{Type: EventThinking, Content: "done", AgentID: work.role.Name, AgentStatus: "done"})

			// 线程安全地累加 token 使用量
			usageMu.Lock()
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
			usageMu.Unlock()

			results[work.idx] = AgentResult{Role: work.role.Name, Content: content}
		}(w)
	}

	// 等待所有代理完成初始回答
	wg.Wait()

	// 筛选有效结果，过滤出错的代理
	var validResults []AgentResult
	var hasError bool
	for _, result := range results {
		if result.Error != nil {
			hasError = true
		} else {
			validResults = append(validResults, result)
		}
	}

	// 所有代理都失败的情况
	if len(validResults) == 0 {
		cb(Event{Type: EventError, Content: "All agents failed to produce a response."})
		cb(Event{Type: EventDone, Usage: UsageInfo{
			PromptTokens:     totalUsage.PromptTokens,
			CompletionTokens: totalUsage.CompletionTokens,
			TotalTokens:      totalUsage.TotalTokens,
		}})
		return "", fmt.Errorf("all agents failed")
	}

	// 部分代理失败时发出警告
	if hasError {
		cb(Event{Type: EventError, Content: "One or more agents encountered an error, but proceeding with available results."})
	}

	// 只有一个有效结果时直接返回，无需辩论
	if len(validResults) == 1 {
		cb(Event{Type: EventAnswer, Content: validResults[0].Content})
		cb(Event{Type: EventDone, Usage: UsageInfo{
			PromptTokens:     totalUsage.PromptTokens,
			CompletionTokens: totalUsage.CompletionTokens,
			TotalTokens:      totalUsage.TotalTokens,
		}})
		return validResults[0].Content, nil
	}

	// === 第二阶段：辩论轮次 ===
	// 每个代理查看其他代理的回答，据此改进自己的答案
	var debateWg sync.WaitGroup
	debateResults := make([]string, len(validResults))

	for i, result := range validResults {
		debateWg.Add(1)
		go func(idx int, r AgentResult) {
			defer debateWg.Done()

			cb(Event{Type: EventThinking, Content: "debating...", AgentID: r.Role, AgentStatus: "debating"})

			// 构建辩论提示词：包含自己的原始回答和其他代理的回答
			var critiquePrompt strings.Builder
			critiquePrompt.WriteString(fmt.Sprintf("You are %s. Here was your original answer:\n\n%s\n\n", r.Role, r.Content))
			critiquePrompt.WriteString("Now here are the answers from other agents:\n\n")
			for _, other := range validResults {
				if other.Role != r.Role {
					critiquePrompt.WriteString(fmt.Sprintf("[%s]:\n%s\n\n", other.Role, other.Content))
				}
			}
			critiquePrompt.WriteString("Review the other answers. If you find valid points you missed, incorporate them. If you disagree, explain why. Produce an improved final answer.")

			debateMessages := []Message{
				{Role: "system", Content: "You are revising your answer after seeing other perspectives. Be thorough and incorporate valid points from others."},
				{Role: "user", Content: critiquePrompt.String()},
			}

			// 辩论阶段使用非流式请求（Chat 而非 ChatStream）
			content, _, usage, err := client.Chat(ctx, debateMessages, nil)
			if err != nil || strings.TrimSpace(content) == "" {
				// 辩论失败时回退到原始回答
				debateResults[idx] = r.Content
				cb(Event{Type: EventThinking, Content: "debate done (fallback)", AgentID: r.Role, AgentStatus: "done"})
				return
			}

			usageMu.Lock()
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
			usageMu.Unlock()

			debateResults[idx] = content
			cb(Event{Type: EventThinking, Content: "debate done", AgentID: r.Role, AgentStatus: "done"})
		}(i, result)
	}

	debateWg.Wait()

	// === 第三阶段：综合所有辩论结果 ===
	// 使用最后一个角色（通常是 Synthesizer）来生成最终综合答案
	synthesizerRole := roles[len(roles)-1]
	cb(Event{Type: EventThinking, Content: "starting...", AgentID: synthesizerRole.Name, AgentStatus: "synthesizing"})

	// 汇总所有代理辩论后的回答
	var debateContent strings.Builder
	for i, result := range validResults {
		debateContent.WriteString(fmt.Sprintf("## [%s] (after debate)\n%s\n\n", result.Role, debateResults[i]))
	}

	// 构建综合提示词
	synthMessages := []Message{
		{Role: "system", Content: synthesizerRole.Prefix + "\n\nYou will be given responses from multiple AI agents who have already debated and refined their positions. Analyze all perspectives, combine the best insights, resolve any contradictions, and produce a single comprehensive answer. Be thorough but concise."},
		{Role: "user", Content: fmt.Sprintf("Original question: %s\n\n%s", userMessage, debateContent.String())},
	}

	toolDefs := buildToolDefs(toolList)

	// 流式生成最终综合答案
	content, _, synthUsage, err := client.ChatStream(ctx, synthMessages, toolDefs, func(chunk string) {
		cb(Event{Type: EventAnswer, Content: chunk})
	}, func(reasoning string) {
		cb(Event{Type: EventThinking, Content: reasoning})
	})
	if err != nil {
		cb(Event{Type: EventError, Content: fmt.Sprintf("Synthesis error: %v", err)})
		return "", fmt.Errorf("synthesis error: %w", err)
	}

	cb(Event{Type: EventThinking, Content: "done", AgentID: synthesizerRole.Name, AgentStatus: "done"})

	// 累加综合阶段的 token 使用量
	totalUsage.PromptTokens += synthUsage.PromptTokens
	totalUsage.CompletionTokens += synthUsage.CompletionTokens
	totalUsage.TotalTokens += synthUsage.TotalTokens

	// 发送完成事件
	cb(Event{Type: EventDone, Usage: UsageInfo{
		PromptTokens:     totalUsage.PromptTokens,
		CompletionTokens: totalUsage.CompletionTokens,
		TotalTokens:      totalUsage.TotalTokens,
	}})
	return content, nil
}
