// web.go 实现了 go-agent 的 Web UI 后端服务器。
// 提供了一个基于 SSE（Server-Sent Events）的聊天 API，
// 以及会话管理、历史记录、连接测试等多个 HTTP 端点。
//
// 主要功能模块：
//   - LLM 提供商管理（支持 Ollama、OpenAI、DeepSeek 等多种服务）
//   - 搜索引擎集成（Tavily、SearXNG）
//   - 实时流式聊天（SSE 协议）
//   - 会话与历史记录管理（JSON 文件持久化）
//   - 静态文件服务（嵌入式前端资源）
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

// staticFS 通过 go:embed 指令将 web/static/ 目录嵌入到二进制文件中。
// 这使得前端静态资源（HTML、CSS、JS）无需外部文件即可部署。
//
//go:embed web/static/*
var staticFS embed.FS

// Provider 定义了一个 LLM 服务提供商的信息。
type Provider struct {
	ID      string `json:"id"`       // 提供商唯一标识（如 "ollama"、"openai"）
	Name    string `json:"name"`     // 显示名称（如 "Ollama (本地)"）
	BaseURL string `json:"base_url"` // API 基础 URL
	Icon    string `json:"icon"`     // 显示图标（emoji）
}

// providers 预定义的 LLM 服务提供商列表，Web UI 前端使用此列表展示可选的服务商。
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

// SearchProvider 定义了一个搜索引擎提供商的信息。
type SearchProvider struct {
	ID      string `json:"id"`       // 提供商唯一标识
	Name    string `json:"name"`     // 显示名称
	BaseURL string `json:"base_url"` // API 基础 URL
	Icon    string `json:"icon"`     // 显示图标
}

// searchProviders 预定义的搜索引擎提供商列表。
var searchProviders = []SearchProvider{
	{ID: "tavily", Name: "Tavily", BaseURL: "https://api.tavily.com", Icon: "🔍"},
	{ID: "searxng", Name: "SearXNG (自建)", BaseURL: "http://localhost:8080", Icon: "🔎"},
}

// Session 表示一个用户的会话状态，包含客户端配置、工具和消息历史。
// 每个 IP 地址对应一个 Session，通过 sync.Map 管理。
type Session struct {
	sessionID     string            // 会话唯一标识
	client        *Client           // LLM 客户端实例
	tools         []Tool            // 当前可用工具列表
	searchTool    *tools.SearchTool // 搜索工具实例（可能为 nil）
	showReasoning bool              // 是否显示模型推理过程
	agentCount    int               // 多代理模式的代理数量
	messages      []Message         // 当前会话的消息历史
	mu            sync.Mutex        // 保护并发访问的互斥锁
}

// ChatSession 表示一个可持久化的聊天会话记录。
// 以 JSON 文件形式存储在 data/history/ 目录中。
type ChatSession struct {
	ID             string    `json:"id"`              // 会话唯一标识
	Title          string    `json:"title"`           // 会话标题（取自第一条用户消息）
	Messages       []Message `json:"messages"`        // 完整的消息列表
	PartialContent string    `json:"partial_content"` // 流式生成中的部分响应（导航中断时保存）
	CreatedAt      time.Time `json:"created_at"`      // 创建时间
	UpdatedAt      time.Time `json:"updated_at"`      // 最后更新时间
}

// HistoryStore 管理聊天历史记录的持久化存储。
// 使用文件系统（JSON 文件）存储会话数据，支持并发读写。
type HistoryStore struct {
	dir string       // 存储目录路径
	mu  sync.RWMutex // 读写锁，允许多个并发读取但互斥写入
}

// NewHistoryStore 创建一个新的历史记录存储实例。
// 参数 dir (string)：存储目录路径，不存在时会自动创建。
// 返回值：*HistoryStore 实例指针。
func NewHistoryStore(dir string) *HistoryStore {
	os.MkdirAll(dir, 0755)
	return &HistoryStore{dir: dir}
}

// Save 保存或更新一个聊天会话记录。
// 参数 session (ChatSession)：要保存的会话数据。
// 文件名格式为 "{session.ID}.json"。
func (hs *HistoryStore) Save(session ChatSession) error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(hs.dir, session.ID+".json"), data, 0644)
}

// List 列出所有聊天会话，按最后更新时间降序排列。
// 返回值：ChatSession 列表和可能的错误。
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
	// 按更新时间降序排列
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})
	return sessions, nil
}

// Load 根据 ID 加载一个聊天会话。
// 参数 id (string)：会话 ID。
// 返回值：ChatSession 指针和可能的错误。
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

// Delete 删除一个聊天会话记录。
// 参数 id (string)：要删除的会话 ID。
func (hs *HistoryStore) Delete(id string) error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	return os.Remove(filepath.Join(hs.dir, id+".json"))
}

// Search 在所有会话中搜索包含指定关键词的会话。
// 参数 query (string)：搜索关键词（不区分大小写）。
// 搜索范围包括会话标题和所有消息内容。
func (hs *HistoryStore) Search(query string) ([]ChatSession, error) {
	all, err := hs.List()
	if err != nil {
		return nil, err
	}
	query = strings.ToLower(query)
	var results []ChatSession
	for _, s := range all {
		// 先搜索标题
		if strings.Contains(strings.ToLower(s.Title), query) {
			results = append(results, s)
			continue
		}
		// 再搜索消息内容
		for _, m := range s.Messages {
			if strings.Contains(strings.ToLower(m.Content), query) {
				results = append(results, s)
				break
			}
		}
	}
	return results, nil
}

// handleChatSSE 处理聊天 SSE 请求，是 Web UI 的核心端点。
// 接收用户消息和配置，通过 SSE 实时返回 AI 回复。
// HTTP 方法：POST
// 请求体 JSON 结构：{ content: 用户消息, config: { provider, base_url, api_key, model, ... } }
// 响应：SSE 格式的事件流，每个事件为一行 "data: {JSON}\n\n"
func handleChatSSE(w http.ResponseWriter, r *http.Request, sessions *sync.Map, history *HistoryStore) {
	if r.Method != "POST" {
		http.Error(w, "POST only", 405)
		return
	}

	// 解析请求体
	var req struct {
		Content string `json:"content"` // 用户消息内容
		Config  struct {
			Provider       string `json:"provider"`        // LLM 提供商 ID
			BaseURL        string `json:"base_url"`        // API 基础 URL
			APIKey         string `json:"api_key"`         // API 密钥
			Model          string `json:"model"`           // 模型名称
			ShowReasoning  bool   `json:"show_reasoning"`  // 是否显示推理过程
			SearchProvider string `json:"search_provider"` // 搜索引擎提供商
			SearchAPIKey   string `json:"search_api_key"`  // 搜索 API 密钥
			SearchBaseURL  string `json:"search_base_url"` // 搜索 API 地址
			SearchEnabled  bool   `json:"search_enabled"`  // 是否启用搜索
			AgentCount     int    `json:"agent_count"`     // 多代理数量
		} `json:"config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	// 根据客户端 IP 获取或创建会话
	sessionKey := strings.Split(r.RemoteAddr, ":")[0]
	sessIface, _ := sessions.LoadOrStore(sessionKey, &Session{
		sessionID: generateSessionID(),
		client:    NewClient(),
		tools:     GetTools(),
		messages:  []Message{},
	})
	sess := sessIface.(*Session)

	// 更新会话配置
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

	// 配置搜索工具
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

	// 设置 SSE 响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// 检查响应写入器是否支持 Flush（SSE 必需）
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	// 复制会话状态用于本次查询
	sess.mu.Lock()
	client := sess.client
	toolList := sess.tools
	agentCount := sess.agentCount
	historyMsgs := make([]Message, len(sess.messages))
	copy(historyMsgs, sess.messages)
	sess.mu.Unlock()

	log.Printf("[sse] starting query: model=%s search=%v agents=%d history=%d",
		client.Model, sess.searchTool != nil, agentCount, len(historyMsgs))

	// eventCh 事件通道，查询 goroutine 产生事件，写入 goroutine 消费事件并发送给客户端
	eventCh := make(chan Event, 1024)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)

	// 写入 goroutine：从 eventCh 读取事件，序列化为 JSON 并写入 SSE 响应
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

	// writeEvent 辅助函数，安全地将事件发送到 eventCh
	writeEvent := func(event Event) {
		select {
		case eventCh <- event:
		case <-ctx.Done():
		}
	}

	// 查询 goroutine：执行实际的 AI 查询（单代理或多代理）
	go func() {
		defer close(eventCh)

		// 将用户消息添加到会话历史
		sess.mu.Lock()
		sess.messages = append(sess.messages, Message{Role: "user", Content: req.Content})
		messages := make([]Message, len(sess.messages))
		copy(messages, sess.messages)
		sessionID := sess.sessionID
		sess.mu.Unlock()

		// 生成会话标题（取第一条用户消息的前 50 字符）
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
			ID:             sessionID,
			Title:          title,
			Messages:       messages,
			PartialContent: "",
			CreatedAt:      now,
			UpdatedAt:      now,
		}
		if saveErr := history.Save(chatSession); saveErr != nil {
			log.Printf("[history] failed to save session: %v", saveErr)
		}

		var assistantContent string
		var err error

		queryCtx, queryCancel := context.WithCancel(r.Context())
		defer queryCancel()

		// 根据代理数量选择单代理或多代理查询
		if agentCount > 1 {
			assistantContent, err = MultiAgentQueryWithCtx(queryCtx, client, toolList, historyMsgs, req.Content, agentCount, writeEvent)
		} else {
			assistantContent, err = QueryWithCallbackAndCtx(queryCtx, client, toolList, historyMsgs, req.Content, writeEvent)
		}

		// 查询成功后更新会话历史
		if err == nil && assistantContent != "" {
			sess.mu.Lock()
			sess.messages = append(sess.messages, Message{Role: "assistant", Content: assistantContent})
			messages = make([]Message, len(sess.messages))
			copy(messages, sess.messages)
			sess.mu.Unlock()

			chatSession.Messages = messages
			chatSession.PartialContent = ""
			chatSession.UpdatedAt = time.Now()
			if saveErr := history.Save(chatSession); saveErr != nil {
				log.Printf("[history] failed to update session: %v", saveErr)
			}
		}
	}()

	wg.Wait()
	log.Printf("[sse] query completed: model=%s agents=%d", client.Model, agentCount)
}

// handleSavePartial 接收前端推送的部分响应内容并保存到历史。
// 当用户在流式生成过程中导航离开时调用。
// HTTP 方法：POST，请求体：{ session_id: 会话 ID, content: 部分响应内容 }
func handleSavePartial(w http.ResponseWriter, r *http.Request, history *HistoryStore) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	var req struct {
		SessionID string `json:"session_id"`
		Content   string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": err.Error()})
		return
	}

	if req.SessionID == "" || req.Content == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "session_id and content required"})
		return
	}

	session, err := history.Load(req.SessionID)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": "session not found"})
		return
	}

	session.PartialContent = req.Content
	session.UpdatedAt = time.Now()
	if err := history.Save(*session); err != nil {
		log.Printf("[partial] failed to save partial content: %v", err)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": false, "message": err.Error()})
		return
	}

	log.Printf("[partial] saved %d chars for session %s", len(req.Content), req.SessionID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

// handleResetSession 重置当前客户端的会话（清空消息历史，生成新会话 ID）。
// HTTP 方法：POST
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

// handleArchiveSession 归档当前会话（保存到历史并清空当前会话）。
// HTTP 方法：POST
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

	// 保存当前会话消息并重置
	sess := sessIface.(*Session)
	sess.mu.Lock()
	messages := make([]Message, len(sess.messages))
	copy(messages, sess.messages)
	oldSessionID := sess.sessionID
	sess.messages = []Message{}
	sess.sessionID = generateSessionID()
	sess.mu.Unlock()

	// 如果有消息则保存到历史记录
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

// handleListHistory 列出所有历史会话的摘要信息（ID、标题、时间）。
// HTTP 方法：GET
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

	// 只返回摘要信息，不包含完整消息列表
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

// handleGetHistory 根据 ID 获取一个历史会话的完整数据。
// HTTP 方法：GET，查询参数：id
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

// handleLoadHistory 加载一个历史会话到当前活动会话中（恢复对话上下文）。
// HTTP 方法：POST，请求体：{ id: 会话 ID }
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

	// 将历史会话的消息恢复到当前活动会话
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

// handleDeleteHistory 删除一个历史会话记录。
// HTTP 方法：DELETE，请求体：{ id: 会话 ID }
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

// handleSearchHistory 根据关键词搜索历史会话。
// HTTP 方法：GET，查询参数：q（搜索关键词）
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

// handleTestConnection 测试与 LLM API 的连接是否正常。
// 通过发送一条简单的 "hi" 消息来验证 API 可用性。
// HTTP 方法：POST，请求体：{ base_url, api_key, model }
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

	// 发送测试消息
	_, _, _, err := client.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
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

// handleGetModels 获取指定 API 端点可用的模型列表。
// 调用 /models 端点获取模型列表并返回模型 ID 数组。
// HTTP 方法：GET，查询参数：base_url, api_key（可选）
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

	// 解析 OpenAI 兼容的模型列表响应
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

	// 提取模型 ID 列表
	ids := make([]string, len(modelsResp.Data))
	for i, m := range modelsResp.Data {
		ids[i] = m.ID
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":     true,
		"models": ids,
	})
}

// handleTestSearch 测试搜索引擎连接是否正常。
// 发送一个 "test" 查询来验证搜索 API 可用性。
// HTTP 方法：POST，请求体：{ provider, api_key, base_url }
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

// min 返回两个整数中较小的那个。
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// generateSessionID 生成一个随机的 32 字符十六进制会话 ID。
// 使用 crypto/rand 生成 16 字节随机数据，然后编码为十六进制字符串。
func generateSessionID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// StartWebServer 启动 Web UI 的 HTTP 服务器。
// 注册所有 API 路由和静态文件服务，然后启动监听。
// 参数 port (int)：服务器监听端口。
// 返回值：服务器停止时返回错误（如监听失败）。
//
// API 路由：
//   - POST /api/chat          - SSE 流式聊天
//   - POST /api/reset         - 重置会话
//   - POST /api/history/archive - 归档会话
//   - GET  /api/history/list  - 列出历史会话
//   - GET  /api/history/get   - 获取指定会话
//   - POST /api/history/load  - 加载历史会话
//   - DELETE /api/history/delete - 删除历史会话
//   - GET  /api/history/search - 搜索历史会话
//   - POST /api/test          - 测试 LLM 连接
//   - GET  /api/models        - 获取可用模型列表
//   - POST /api/test-search   - 测试搜索引擎连接
//   - GET  /api/providers     - 获取支持的提供商列表
//   - GET  /static/*          - 静态文件服务
//   - GET  /                  - Web UI 首页
func StartWebServer(port int) error {
	sessions := &sync.Map{}
	history := NewHistoryStore("data/history")

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		handleChatSSE(w, r, sessions, history)
	})
	mux.HandleFunc("/api/chat/save-partial", func(w http.ResponseWriter, r *http.Request) {
		handleSavePartial(w, r, history)
	})
	mux.HandleFunc("/api/reset", func(w http.ResponseWriter, r *http.Request) {
		handleResetSession(w, r, sessions)
	})
	mux.HandleFunc("/api/history/archive", func(w http.ResponseWriter, r *http.Request) {
		handleArchiveSession(w, r, sessions, history)
	})
	mux.HandleFunc("/api/history/list", func(w http.ResponseWriter, r *http.Request) {
		handleListHistory(w, r, history)
	})
	mux.HandleFunc("/api/history/get", func(w http.ResponseWriter, r *http.Request) {
		handleGetHistory(w, r, history)
	})
	mux.HandleFunc("/api/history/load", func(w http.ResponseWriter, r *http.Request) {
		handleLoadHistory(w, r, sessions, history)
	})
	mux.HandleFunc("/api/history/delete", func(w http.ResponseWriter, r *http.Request) {
		handleDeleteHistory(w, r, history)
	})
	mux.HandleFunc("/api/history/search", func(w http.ResponseWriter, r *http.Request) {
		handleSearchHistory(w, r, history)
	})
	mux.HandleFunc("/api/test", handleTestConnection)
	mux.HandleFunc("/api/models", handleGetModels)
	mux.HandleFunc("/api/test-search", handleTestSearch)
	// 返回所有支持的 LLM 和搜索引擎提供商列表
	mux.HandleFunc("/api/providers", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"llm":    providers,
			"search": searchProviders,
		})
	})
	// 静态文件服务：从嵌入的文件系统提供 JS、CSS 等静态资源
	mux.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
		filePath := "web" + r.URL.Path
		data, err := staticFS.ReadFile(filePath)
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
		// 根据文件扩展名设置正确的 Content-Type
		if strings.HasSuffix(r.URL.Path, ".js") {
			w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		} else if strings.HasSuffix(r.URL.Path, ".css") {
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
		}
		// 静态资源缓存 1 天
		w.Header().Set("Cache-Control", "public, max-age=86400")
		w.Write(data)
	})
	// 首页：返回 index.html
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, err := staticFS.ReadFile("web/static/index.html")
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		// HTML 不缓存，确保始终获取最新版本
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Write(data)
	})

	// 配置并启动 HTTP 服务器
	addr := fmt.Sprintf(":%d", port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,  // 读取请求超时
		WriteTimeout: 120 * time.Second, // 写入响应超时（SSE 需要较长时间）
		IdleTimeout:  120 * time.Second, // 空闲连接超时
	}

	log.Printf("🌐 go-agent web UI running at http://localhost%s", addr)
	return srv.ListenAndServe()
}
