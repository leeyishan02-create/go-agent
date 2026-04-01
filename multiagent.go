package main

import (
	"fmt"
	"strings"
	"sync"
)

type AgentRole struct {
	Name   string
	Prefix string
}

var defaultRoles = []AgentRole{
	{Name: "Analyst", Prefix: "You are an analytical assistant. Break down the problem, examine it from multiple angles, and provide a thorough analysis."},
	{Name: "Researcher", Prefix: "You are a research-focused assistant. Gather relevant information, consider edge cases, and provide well-researched insights."},
	{Name: "Reviewer", Prefix: "You are a critical reviewer. Evaluate the problem carefully, identify potential issues, and suggest improvements."},
	{Name: "Synthesizer", Prefix: "You are a synthesis expert. Your job is to combine multiple perspectives into a single, coherent, and comprehensive answer."},
}

type AgentResult struct {
	Role    string
	Content string
	Error   error
}

func MultiAgentQuery(client *Client, toolList []Tool, userMessage string, agentCount int, cb EventCallback) error {
	_, err := MultiAgentQueryWithHistory(client, toolList, nil, userMessage, agentCount, cb)
	return err
}

func MultiAgentQueryWithHistory(client *Client, toolList []Tool, history []Message, userMessage string, agentCount int, cb EventCallback) (string, error) {
	if agentCount < 1 {
		agentCount = 1
	}

	roles := defaultRoles[:min(agentCount, len(defaultRoles))]
	if agentCount > len(defaultRoles) {
		for i := len(defaultRoles); i < agentCount; i++ {
			roles = append(roles, AgentRole{
				Name:   fmt.Sprintf("Agent-%d", i+1),
				Prefix: fmt.Sprintf("You are assistant #%d. Provide a unique and thoughtful perspective on the problem.", i+1),
			})
		}
	}

	var wg sync.WaitGroup
	results := make([]AgentResult, len(roles))
	var usageMu sync.Mutex
	var totalUsage Usage

	for i, role := range roles {
		wg.Add(1)
		go func(idx int, r AgentRole) {
			defer wg.Done()

			var messages []Message
			messages = append(messages, Message{Role: "system", Content: r.Prefix + "\n\n" + buildSystemPrompt(toolList)})
			messages = append(messages, history...)
			messages = append(messages, Message{Role: "user", Content: userMessage})

			toolDefs := buildToolDefs(toolList)

			cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] starting...", r.Name), AgentID: r.Name, AgentStatus: "thinking"})

			content, _, usage, err := client.ChatStream(messages, toolDefs, nil, nil)
			if err != nil {
				cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] error: %v", r.Name, err), AgentID: r.Name, AgentStatus: "error"})
				results[idx] = AgentResult{Role: r.Name, Error: err}
				return
			}

			if strings.TrimSpace(content) == "" {
				cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] empty response", r.Name), AgentID: r.Name, AgentStatus: "error"})
				results[idx] = AgentResult{Role: r.Name, Error: fmt.Errorf("empty response")}
				return
			}

			cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] done", r.Name), AgentID: r.Name, AgentStatus: "done"})

			usageMu.Lock()
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
			usageMu.Unlock()

			results[idx] = AgentResult{Role: r.Name, Content: content}
		}(i, role)
	}

	wg.Wait()

	var validResults []AgentResult
	var hasError bool
	for _, result := range results {
		if result.Error != nil {
			hasError = true
		} else {
			validResults = append(validResults, result)
		}
	}

	if len(validResults) == 0 {
		cb(Event{Type: EventError, Content: "All agents failed to produce a response."})
		cb(Event{Type: EventDone, Usage: UsageInfo{
			PromptTokens:     totalUsage.PromptTokens,
			CompletionTokens: totalUsage.CompletionTokens,
			TotalTokens:      totalUsage.TotalTokens,
		}})
		return "", fmt.Errorf("all agents failed")
	}

	if hasError {
		cb(Event{Type: EventError, Content: "One or more agents encountered an error, but proceeding with available results."})
	}

	if len(validResults) == 1 {
		cb(Event{Type: EventAnswer, Content: validResults[0].Content})
		cb(Event{Type: EventDone, Usage: UsageInfo{
			PromptTokens:     totalUsage.PromptTokens,
			CompletionTokens: totalUsage.CompletionTokens,
			TotalTokens:      totalUsage.TotalTokens,
		}})
		return validResults[0].Content, nil
	}

	// Debate round
	cb(Event{Type: EventThinking, Content: "Starting debate round...", AgentStatus: "debate_start"})

	var debateWg sync.WaitGroup
	debateResults := make([]string, len(validResults))

	for i, result := range validResults {
		debateWg.Add(1)
		go func(idx int, r AgentResult) {
			defer debateWg.Done()

			cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] debating...", r.Role), AgentID: r.Role, AgentStatus: "debating"})

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

			content, _, usage, err := client.Chat(debateMessages, nil)
			if err != nil || strings.TrimSpace(content) == "" {
				debateResults[idx] = r.Content
				cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] debate done (fallback)", r.Role), AgentID: r.Role, AgentStatus: "done"})
				return
			}

			usageMu.Lock()
			totalUsage.PromptTokens += usage.PromptTokens
			totalUsage.CompletionTokens += usage.CompletionTokens
			totalUsage.TotalTokens += usage.TotalTokens
			usageMu.Unlock()

			debateResults[idx] = content
			cb(Event{Type: EventThinking, Content: fmt.Sprintf("[%s] debate done", r.Role), AgentID: r.Role, AgentStatus: "done"})
		}(i, result)
	}

	debateWg.Wait()

	cb(Event{Type: EventThinking, Content: "Synthesizing final answer...", AgentStatus: "synthesizing"})

	synthesizerRole := roles[len(roles)-1]
	var debateContent strings.Builder
	for i, result := range validResults {
		debateContent.WriteString(fmt.Sprintf("## [%s] (after debate)\n%s\n\n", result.Role, debateResults[i]))
	}

	synthMessages := []Message{
		{Role: "system", Content: synthesizerRole.Prefix + "\n\nYou will be given responses from multiple AI agents who have already debated and refined their positions. Analyze all perspectives, combine the best insights, resolve any contradictions, and produce a single comprehensive answer. Be thorough but concise."},
		{Role: "user", Content: fmt.Sprintf("Original question: %s\n\n%s", userMessage, debateContent.String())},
	}

	toolDefs := buildToolDefs(toolList)

	content, _, synthUsage, err := client.ChatStream(synthMessages, toolDefs, func(chunk string) {
		cb(Event{Type: EventAnswer, Content: chunk})
	}, func(reasoning string) {
		cb(Event{Type: EventThinking, Content: reasoning})
	})
	if err != nil {
		cb(Event{Type: EventError, Content: fmt.Sprintf("Synthesis error: %v", err)})
		return "", fmt.Errorf("synthesis error: %w", err)
	}

	totalUsage.PromptTokens += synthUsage.PromptTokens
	totalUsage.CompletionTokens += synthUsage.CompletionTokens
	totalUsage.TotalTokens += synthUsage.TotalTokens

	cb(Event{Type: EventDone, Usage: UsageInfo{
		PromptTokens:     totalUsage.PromptTokens,
		CompletionTokens: totalUsage.CompletionTokens,
		TotalTokens:      totalUsage.TotalTokens,
	}})
	return content, nil
}
