package main

import (
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

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
	sessionID     string
	client        *Client
	tools         []Tool
	searchTool    *tools.SearchTool
	showReasoning bool
	agentCount    int
	messages      []Message
	mu            sync.Mutex
}

type ChatSession struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Messages  []Message `json:"messages"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type HistoryStore struct {
	dir string
	mu  sync.RWMutex
}

func NewHistoryStore(dir string) *HistoryStore {
	os.MkdirAll(dir, 0755)
	return &HistoryStore{dir: dir}
}

func (hs *HistoryStore) Save(session ChatSession) error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(hs.dir, session.ID+".json"), data, 0644)
}

func (hs *HistoryStore) List() ([]ChatSession, error) {
	hs.mu.RLock()
	defer hs.mu.RUnlock()
	entries, err := os.ReadDir(hs.dir)
	if err != nil {
		return nil, err
	}
	var sessions []ChatSession
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(hs.dir, e.Name()))
		if err != nil {
			continue
		}
		var s ChatSession
		if err := json.Unmarshal(data, &s); err != nil {
			continue
		}
		sessions = append(sessions, s)
	}
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})
	return sessions, nil
}

func (hs *HistoryStore) Load(id string) (*ChatSession, error) {
	hs.mu.RLock()
	defer hs.mu.RUnlock()
	data, err := os.ReadFile(filepath.Join(hs.dir, id+".json"))
	if err != nil {
		return nil, err
	}
	var s ChatSession
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

func (hs *HistoryStore) Delete(id string) error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	return os.Remove(filepath.Join(hs.dir, id+".json"))
}

func (hs *HistoryStore) Search(query string) ([]ChatSession, error) {
	all, err := hs.List()
	if err != nil {
		return nil, err
	}
	query = strings.ToLower(query)
	var results []ChatSession
	for _, s := range all {
		if strings.Contains(strings.ToLower(s.Title), query) {
			results = append(results, s)
			continue
		}
		for _, m := range s.Messages {
			if strings.Contains(strings.ToLower(m.Content), query) {
				results = append(results, s)
				break
			}
		}
	}
	return results, nil
}

func handleChatSSE(w http.ResponseWriter, r *http.Request, sessions *sync.Map, history *HistoryStore) {
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

	sessionKey := strings.Split(r.RemoteAddr, ":")[0]
	sessIface, _ := sessions.LoadOrStore(sessionKey, &Session{
		sessionID: generateSessionID(),
		client:    NewClient(),
		tools:     GetTools(),
		messages:  []Message{},
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
	historyMsgs := make([]Message, len(sess.messages))
	copy(historyMsgs, sess.messages)
	sess.mu.Unlock()

	log.Printf("[sse] starting query: model=%s search=%v agents=%d history=%d",
		client.Model, sess.searchTool != nil, agentCount, len(historyMsgs))

	eventCh := make(chan Event, 1024)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)

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

		sess.mu.Lock()
		sess.messages = append(sess.messages, Message{Role: "user", Content: req.Content})
		messages := make([]Message, len(sess.messages))
		copy(messages, sess.messages)
		sessionID := sess.sessionID
		sess.mu.Unlock()

		title := "新对话"
		for _, m := range messages {
			if m.Role == "user" && m.Content != "" {
				if len(m.Content) > 50 {
					title = m.Content[:50] + "..."
				} else {
					title = m.Content
				}
				break
			}
		}
		now := time.Now()
		chatSession := ChatSession{
			ID:        sessionID,
			Title:     title,
			Messages:  messages,
			CreatedAt: now,
			UpdatedAt: now,
		}
		if saveErr := history.Save(chatSession); saveErr != nil {
			log.Printf("[history] failed to save session: %v", saveErr)
		}

		var assistantContent string
		var err error

		if agentCount > 1 {
			assistantContent, err = MultiAgentQueryWithHistory(client, toolList, historyMsgs, req.Content, agentCount, writeEvent)
		} else {
			assistantContent, err = QueryWithCallbackAndHistory(client, toolList, historyMsgs, req.Content, writeEvent)
		}

		if err == nil && assistantContent != "" {
			sess.mu.Lock()
			sess.messages = append(sess.messages, Message{Role: "assistant", Content: assistantContent})
			messages = make([]Message, len(sess.messages))
			copy(messages, sess.messages)
			sess.mu.Unlock()

			chatSession.Messages = messages
			chatSession.UpdatedAt = time.Now()
			if saveErr := history.Save(chatSession); saveErr != nil {
				log.Printf("[history] failed to update session: %v", saveErr)
			}
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

	sessionKey := strings.Split(r.RemoteAddr, ":")[0]
	sessIface, ok := sessions.Load(sessionKey)
	if ok {
		sess := sessIface.(*Session)
		sess.mu.Lock()
		sess.messages = []Message{}
		sess.sessionID = generateSessionID()
		sess.mu.Unlock()
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

func handleArchiveSession(w http.ResponseWriter, r *http.Request, sessions *sync.Map, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	sessionKey := strings.Split(r.RemoteAddr, ":")[0]
	sessIface, ok := sessions.Load(sessionKey)
	if !ok {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
		return
	}

	sess := sessIface.(*Session)
	sess.mu.Lock()
	messages := make([]Message, len(sess.messages))
	copy(messages, sess.messages)
	oldSessionID := sess.sessionID
	sess.messages = []Message{}
	sess.sessionID = generateSessionID()
	sess.mu.Unlock()

	if len(messages) > 0 {
		title := "新对话"
		for _, m := range messages {
			if m.Role == "user" && m.Content != "" {
				if len(m.Content) > 50 {
					title = m.Content[:50] + "..."
				} else {
					title = m.Content
				}
				break
			}
		}
		now := time.Now()
		chatSession := ChatSession{
			ID:        oldSessionID,
			Title:     title,
			Messages:  messages,
			CreatedAt: now,
			UpdatedAt: now,
		}
		if err := history.Save(chatSession); err != nil {
			log.Printf("[history] failed to save session: %v", err)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

func handleListHistory(w http.ResponseWriter, r *http.Request, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	sessions, err := history.List()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": err.Error()})
		return
	}

	type SessionSummary struct {
		ID        string    `json:"id"`
		Title     string    `json:"title"`
		CreatedAt time.Time `json:"created_at"`
		UpdatedAt time.Time `json:"updated_at"`
	}
	summaries := make([]SessionSummary, len(sessions))
	for i, s := range sessions {
		summaries[i] = SessionSummary{
			ID:        s.ID,
			Title:     s.Title,
			CreatedAt: s.CreatedAt,
			UpdatedAt: s.UpdatedAt,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "sessions": summaries})
}

func handleGetHistory(w http.ResponseWriter, r *http.Request, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	id := r.URL.Query().Get("id")
	if id == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "id required"})
		return
	}

	session, err := history.Load(id)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "session not found"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "session": session})
}

func handleLoadHistory(w http.ResponseWriter, r *http.Request, sessions *sync.Map, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	session, err := history.Load(req.ID)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "session not found"})
		return
	}

	sessionKey := strings.Split(r.RemoteAddr, ":")[0]
	sessIface, _ := sessions.LoadOrStore(sessionKey, &Session{
		sessionID: generateSessionID(),
		client:    NewClient(),
		tools:     GetTools(),
		messages:  []Message{},
	})
	sess := sessIface.(*Session)
	sess.mu.Lock()
	sess.messages = session.Messages
	sess.sessionID = session.ID
	sess.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

func handleDeleteHistory(w http.ResponseWriter, r *http.Request, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	if err := history.Delete(req.ID); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": err.Error()})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

func handleSearchHistory(w http.ResponseWriter, r *http.Request, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	query := r.URL.Query().Get("q")
	if query == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "query required"})
		return
	}

	sessions, err := history.Search(query)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": err.Error()})
		return
	}

	type SessionSummary struct {
		ID        string    `json:"id"`
		Title     string    `json:"title"`
		CreatedAt time.Time `json:"created_at"`
		UpdatedAt time.Time `json:"updated_at"`
	}
	summaries := make([]SessionSummary, len(sessions))
	for i, s := range sessions {
		summaries[i] = SessionSummary{
			ID:        s.ID,
			Title:     s.Title,
			CreatedAt: s.CreatedAt,
			UpdatedAt: s.UpdatedAt,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "sessions": summaries})
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

func generateSessionID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func StartWebServer(port int) error {
	sessions := &sync.Map{}
	history := NewHistoryStore("data/history")

	http.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		handleChatSSE(w, r, sessions, history)
	})
	http.HandleFunc("/api/reset", func(w http.ResponseWriter, r *http.Request) {
		handleResetSession(w, r, sessions)
	})
	http.HandleFunc("/api/history/archive", func(w http.ResponseWriter, r *http.Request) {
		handleArchiveSession(w, r, sessions, history)
	})
	http.HandleFunc("/api/history/list", func(w http.ResponseWriter, r *http.Request) {
		handleListHistory(w, r, history)
	})
	http.HandleFunc("/api/history/get", func(w http.ResponseWriter, r *http.Request) {
		handleGetHistory(w, r, history)
	})
	http.HandleFunc("/api/history/load", func(w http.ResponseWriter, r *http.Request) {
		handleLoadHistory(w, r, sessions, history)
	})
	http.HandleFunc("/api/history/delete", func(w http.ResponseWriter, r *http.Request) {
		handleDeleteHistory(w, r, history)
	})
	http.HandleFunc("/api/history/search", func(w http.ResponseWriter, r *http.Request) {
		handleSearchHistory(w, r, history)
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
		filePath := "web" + r.URL.Path
		data, err := staticFS.ReadFile(filePath)
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
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
