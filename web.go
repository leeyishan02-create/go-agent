package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"

	"go-agent/tools"
)

//go:embed web/static/*
var staticFS embed.FS

type Provider struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	BaseURL string `json:"base_url"`
	Icon    string `json:"icon"`
}

var providers = []Provider{
	{ID: "ollama", Name: "Ollama (本地)", BaseURL: "http://localhost:11434/v1", Icon: "🦙"},
	{ID: "openrouter", Name: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", Icon: "🌐"},
	{ID: "openai", Name: "OpenAI", BaseURL: "https://api.openai.com/v1", Icon: "🔵"},
	{ID: "deepseek", Name: "DeepSeek", BaseURL: "https://api.deepseek.com/v1", Icon: "🟢"},
	{ID: "siliconflow", Name: "SiliconFlow (硅基流动)", BaseURL: "https://api.siliconflow.cn/v1", Icon: "🟠"},
	{ID: "dashscope", Name: "DashScope (通义千问)", BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", Icon: "🔴"},
	{ID: "zhipuai", Name: "ZhipuAI (智谱清言)", BaseURL: "https://open.bigmodel.cn/api/paas/v4", Icon: "🟣"},
	{ID: "moonshot", Name: "Moonshot (Kimi)", BaseURL: "https://api.moonshot.cn/v1", Icon: "⚪"},
	{ID: "minimax", Name: "MiniMax (海螺AI)", BaseURL: "https://api.minimax.chat/v1", Icon: "🔶"},
	{ID: "custom", Name: "Custom (自定义)", BaseURL: "", Icon: "🟡"},
}

type SearchProvider struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	BaseURL string `json:"base_url"`
	Icon    string `json:"icon"`
}

var searchProviders = []SearchProvider{
	{ID: "tavily", Name: "Tavily", BaseURL: "https://api.tavily.com", Icon: "🔍"},
	{ID: "searxng", Name: "SearXNG (自建)", BaseURL: "http://localhost:8080", Icon: "🔎"},
}

type Session struct {
	client        *Client
	tools         []Tool
	searchTool    *tools.SearchTool
	showReasoning bool
	agentCount    int
	messages      []Message
	mu            sync.Mutex
}

func handleChatSSE(w http.ResponseWriter, r *http.Request, sessions *sync.Map) {
	if r.Method != "POST" {
		http.Error(w, "POST only", 405)
		return
	}

	var req struct {
		Content string `json:"content"`
		Config  struct {
			Provider       string `json:"provider"`
			BaseURL        string `json:"base_url"`
			APIKey         string `json:"api_key"`
			Model          string `json:"model"`
			ShowReasoning  bool   `json:"show_reasoning"`
			SearchProvider string `json:"search_provider"`
			SearchAPIKey   string `json:"search_api_key"`
			SearchBaseURL  string `json:"search_base_url"`
			SearchEnabled  bool   `json:"search_enabled"`
			AgentCount     int    `json:"agent_count"`
		} `json:"config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	sessionKey := r.RemoteAddr
	sessIface, _ := sessions.LoadOrStore(sessionKey, &Session{
		client:   NewClient(),
		tools:    GetTools(),
		messages: []Message{},
	})
	sess := sessIface.(*Session)

	sess.mu.Lock()
	baseURL := req.Config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	sess.client = &Client{
		BaseURL: baseURL,
		APIKey:  req.Config.APIKey,
		Model:   req.Config.Model,
	}
	sess.showReasoning = req.Config.ShowReasoning
	sess.agentCount = req.Config.AgentCount
	if sess.agentCount < 1 {
		sess.agentCount = 1
	}

	if req.Config.SearchProvider != "" && req.Config.SearchAPIKey != "" && req.Config.SearchEnabled {
		sess.searchTool = &tools.SearchTool{
			Provider: req.Config.SearchProvider,
			APIKey:   req.Config.SearchAPIKey,
			BaseURL:  req.Config.SearchBaseURL,
		}
		sess.tools = GetToolsWithSearch(sess.searchTool)
	} else {
		sess.searchTool = nil
		sess.tools = GetTools()
	}
	sess.mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	sess.mu.Lock()
	client := sess.client
	toolList := sess.tools
	agentCount := sess.agentCount
	history := make([]Message, len(sess.messages))
	copy(history, sess.messages)
	sess.mu.Unlock()

	log.Printf("[sse] starting query: model=%s search=%v agents=%d history=%d",
		client.Model, sess.searchTool != nil, agentCount, len(history))

	eventCh := make(chan Event, 1024)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)

	// Single goroutine writes to HTTP response
	go func() {
		defer wg.Done()
		defer cancel()
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[sse] write goroutine panic: %v", r)
			}
		}()
		for event := range eventCh {
			data, _ := json.Marshal(event)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			if event.Type == EventDone {
				return
			}
		}
	}()

	writeEvent := func(event Event) {
		select {
		case eventCh <- event:
		case <-ctx.Done():
		}
	}

	go func() {
		defer close(eventCh)
		var assistantContent string
		var err error

		if agentCount > 1 {
			assistantContent, err = MultiAgentQueryWithHistory(client, toolList, history, req.Content, agentCount, writeEvent)
		} else {
			assistantContent, err = QueryWithCallbackAndHistory(client, toolList, history, req.Content, writeEvent)
		}

		if err == nil && assistantContent != "" {
			sess.mu.Lock()
			sess.messages = append(sess.messages, Message{Role: "user", Content: req.Content})
			sess.messages = append(sess.messages, Message{Role: "assistant", Content: assistantContent})
			sess.mu.Unlock()
		}
	}()

	wg.Wait()
	log.Printf("[sse] query completed: model=%s agents=%d", client.Model, agentCount)
}

func handleResetSession(w http.ResponseWriter, r *http.Request, sessions *sync.Map) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	sessionKey := r.RemoteAddr
	sessIface, ok := sessions.Load(sessionKey)
	if ok {
		sess := sessIface.(*Session)
		sess.mu.Lock()
		sess.messages = []Message{}
		sess.mu.Unlock()
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

func handleTestConnection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	var req struct {
		BaseURL string `json:"base_url"`
		APIKey  string `json:"api_key"`
		Model   string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	client := &Client{
		BaseURL: req.BaseURL,
		APIKey:  req.APIKey,
		Model:   req.Model,
	}

	_, _, _, err := client.Chat([]Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":      false,
			"message": err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":      true,
		"message": "Connection successful",
	})
}

func handleGetModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	baseURL := r.URL.Query().Get("base_url")
	apiKey := r.URL.Query().Get("api_key")
	if baseURL == "" {
		http.Error(w, "base_url required", 400)
		return
	}

	client := &http.Client{}
	req, _ := http.NewRequest("GET", strings.TrimRight(baseURL, "/")+"/models", nil)
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":      false,
			"message": err.Error(),
		})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	w.Header().Set("Content-Type", "application/json")
	if resp.StatusCode != 200 {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":      false,
			"message": fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body)),
		})
		return
	}

	var modelsResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &modelsResp); err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":      false,
			"message": "failed to parse models: " + err.Error(),
		})
		return
	}

	ids := make([]string, len(modelsResp.Data))
	for i, m := range modelsResp.Data {
		ids[i] = m.ID
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":     true,
		"models": ids,
	})
}

func handleTestSearch(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	var req struct {
		Provider string `json:"provider"`
		APIKey   string `json:"api_key"`
		BaseURL  string `json:"base_url"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	tool := &tools.SearchTool{
		Provider: req.Provider,
		APIKey:   req.APIKey,
		BaseURL:  req.BaseURL,
	}
	result, err := tool.Execute(json.RawMessage(`{"query":"test"}`))
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":      false,
			"message": err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":      true,
		"message": "Search test successful",
		"result":  result[:min(200, len(result))],
	})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func StartWebServer(port int) error {
	sessions := &sync.Map{}

	http.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		handleChatSSE(w, r, sessions)
	})
	http.HandleFunc("/api/reset", func(w http.ResponseWriter, r *http.Request) {
		handleResetSession(w, r, sessions)
	})
	http.HandleFunc("/api/test", handleTestConnection)
	http.HandleFunc("/api/models", handleGetModels)
	http.HandleFunc("/api/test-search", handleTestSearch)
	http.HandleFunc("/api/providers", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"llm":    providers,
			"search": searchProviders,
		})
	})
	http.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
		// Serve static files from embedded filesystem
		// Request path "/static/foo.js" maps to "web/static/foo.js" in embed
		filePath := "web" + r.URL.Path
		data, err := staticFS.ReadFile(filePath)
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
		// Set content type based on extension
		if strings.HasSuffix(r.URL.Path, ".js") {
			w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		} else if strings.HasSuffix(r.URL.Path, ".css") {
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
		}
		w.Header().Set("Cache-Control", "public, max-age=86400")
		w.Write(data)
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, err := staticFS.ReadFile("web/static/index.html")
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Write(data)
	})

	addr := fmt.Sprintf(":%d", port)
	log.Printf("🌐 go-agent web UI running at http://localhost%s", addr)
	return http.ListenAndServe(addr, nil)
}
