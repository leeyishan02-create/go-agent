// search.go 实现了网络搜索工具（SearchTool），允许 AI 代理搜索互联网获取最新信息。
// 支持两种搜索引擎后端：
//   - Tavily：商业搜索 API 服务（https://api.tavily.com）
//   - SearXNG：开源的元搜索引擎（可自建部署）
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// searchHTTPClient 是搜索工具专用的 HTTP 客户端。
// 与主 API 客户端分开配置，使用更短的超时时间和更少的连接池资源。
// 配置参数：
//   - Timeout: 30 秒总超时
//   - MaxIdleConns: 最多保持 10 个空闲连接
//   - MaxIdleConnsPerHost: 每个主机最多 5 个空闲连接
//   - IdleConnTimeout: 空闲连接 30 秒后关闭
var searchHTTPClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout:     30 * time.Second,
	},
}

// SearchTool 是网络搜索工具的实现，实现了 Tool 接口。
// 通过外部搜索引擎 API（Tavily 或 SearXNG）执行网络搜索。
type SearchTool struct {
	// Provider 搜索引擎提供商标识（string 类型），支持 "tavily" 或 "searxng"
	Provider string
	// APIKey 搜索引擎 API 密钥（string 类型），Tavily 必须提供，SearXNG 可选
	APIKey string
	// BaseURL 搜索引擎 API 基础 URL（string 类型），SearXNG 使用此字段指定自建实例地址
	BaseURL string
}

// SearchInput 定义了搜索工具的输入参数结构。
type SearchInput struct {
	// Query 搜索查询关键词（string 类型），作为搜索词发送给搜索引擎
	Query string `json:"query"`
}

// Name 返回工具的名称标识符 "search"。
// 返回值：string 类型的工具名称。
func (t *SearchTool) Name() string {
	return "search"
}

// Description 返回工具的功能描述。
// 返回值：string 类型的工具描述，告知 AI 模型该工具用于获取最新的网络信息。
func (t *SearchTool) Description() string {
	return "Search the web for current information. Use this when you need up-to-date information that may not be in your training data."
}

// InputSchema 返回工具输入参数的 JSON Schema 定义。
// 返回值：map[string]interface{} 类型的 JSON Schema，定义了以下参数：
//   - query (string, 必填)：搜索查询关键词
func (t *SearchTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query",
			},
		},
		"required": []string{"query"},
	}
}

// Execute 执行网络搜索操作。
// 参数：
//   - input (json.RawMessage)：JSON 格式的输入参数，包含 "query" 字段
//
// 返回值：
//   - string：搜索结果的格式化文本，每条结果包含序号、标题、URL 和摘要
//   - error：搜索失败时返回错误信息（如 API 密钥未配置、未知提供商等）
//
// 执行流程：
//  1. 解析 JSON 输入参数
//  2. 验证 query 和 API 密钥
//  3. 根据 Provider 字段分发到对应的搜索引擎后端
func (t *SearchTool) Execute(input json.RawMessage) (string, error) {
	// 将 JSON 输入解析为 SearchInput 结构体
	var args SearchInput
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	// 验证搜索关键词是否为空
	if args.Query == "" {
		return "", fmt.Errorf("query is required")
	}

	// 验证 API 密钥是否已配置
	if t.APIKey == "" {
		return "", fmt.Errorf("search API key not configured")
	}

	// 根据搜索引擎提供商分发请求
	switch t.Provider {
	case "tavily":
		return t.searchTavily(context.Background(), args.Query)
	case "searxng":
		return t.searchSearxNG(context.Background(), args.Query)
	default:
		return "", fmt.Errorf("unknown search provider: %s", t.Provider)
	}
}

// searchTavily 使用 Tavily API 执行网络搜索。
// Tavily 是一个专为 AI 代理设计的搜索 API 服务。
// 参数：
//   - ctx (context.Context)：用于控制请求超时和取消的上下文
//   - query (string)：搜索关键词
//
// 返回值：
//   - string：格式化的搜索结果列表，每条包含 [序号] 标题、URL、内容摘要
//   - error：请求失败或解析失败时返回错误
//
// API 请求参数：
//   - search_depth: "basic"（基本搜索深度）
//   - max_results: 5（最多返回 5 条结果）
func (t *SearchTool) searchTavily(ctx context.Context, query string) (string, error) {
	// 构建 Tavily API 请求体
	body := map[string]interface{}{
		"api_key":      t.APIKey,
		"query":        query,
		"search_depth": "basic",
		"max_results":  5,
	}
	// bodyBytes ([]byte)：序列化后的 JSON 请求数据
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal tavily request: %w", err)
	}

	// 创建带上下文的 HTTP POST 请求
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.tavily.com/search", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create tavily request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// 发送请求
	resp, err := searchHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("tavily request: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应体
	// respBody ([]byte)：API 返回的原始响应数据
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read tavily response: %w", err)
	}

	// 检查 HTTP 状态码
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("tavily API error %d: %s", resp.StatusCode, string(respBody))
	}

	// 解析 Tavily API 的 JSON 响应
	// result.Results 包含搜索结果数组，每个结果有：
	//   - Title (string)：结果标题
	//   - URL (string)：结果链接
	//   - Content (string)：内容摘要
	//   - Score (float64)：相关性评分
	var result struct {
		Results []struct {
			Title   string  `json:"title"`
			URL     string  `json:"url"`
			Content string  `json:"content"`
			Score   float64 `json:"score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse tavily response: %w", err)
	}

	// 如果没有搜索结果，返回提示信息
	if len(result.Results) == 0 {
		return "No results found.", nil
	}

	// 格式化搜索结果为可读文本
	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	return sb.String(), nil
}

// searchSearxNG 使用 SearXNG 实例执行网络搜索。
// SearXNG 是一个开源的隐私保护型元搜索引擎，支持自建部署。
// 参数：
//   - ctx (context.Context)：用于控制请求超时和取消的上下文
//   - query (string)：搜索关键词
//
// 返回值：
//   - string：格式化的搜索结果列表
//   - error：请求失败或解析失败时返回错误
func (t *SearchTool) searchSearxNG(ctx context.Context, query string) (string, error) {
	// 确定 SearXNG 实例的基础 URL，默认为本地 8080 端口
	baseURL := t.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	// 构建搜索 URL，使用 JSON 格式返回，搜索分类为 "general"
	// url.QueryEscape 对查询关键词进行 URL 编码
	searchURL := fmt.Sprintf("%s/search?q=%s&format=json&categories=general",
		baseURL, url.QueryEscape(query))

	// 创建带上下文的 HTTP GET 请求
	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return "", fmt.Errorf("create searxng request: %w", err)
	}

	// 发送请求
	resp, err := searchHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("searxng request: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应体
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read searxng response: %w", err)
	}

	// 检查 HTTP 状态码
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("searxng API error %d: %s", resp.StatusCode, string(respBody))
	}

	// 解析 SearXNG 的 JSON 响应
	// result.Results 包含搜索结果数组，每个结果有：
	//   - Title (string)：结果标题
	//   - URL (string)：结果链接
	//   - Content (string)：内容摘要
	var result struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse searxng response: %w", err)
	}

	// 如果没有搜索结果，返回提示信息
	if len(result.Results) == 0 {
		return "No results found.", nil
	}

	// 格式化搜索结果为可读文本
	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	return sb.String(), nil
}
