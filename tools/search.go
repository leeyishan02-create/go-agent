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

var searchHTTPClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout:     30 * time.Second,
	},
}

type SearchTool struct {
	Provider string
	APIKey   string
	BaseURL  string
}

type SearchInput struct {
	Query string `json:"query"`
}

func (t *SearchTool) Name() string {
	return "search"
}

func (t *SearchTool) Description() string {
	return "Search the web for current information. Use this when you need up-to-date information that may not be in your training data."
}

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

func (t *SearchTool) Execute(input json.RawMessage) (string, error) {
	var args SearchInput
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	if args.Query == "" {
		return "", fmt.Errorf("query is required")
	}

	if t.APIKey == "" {
		return "", fmt.Errorf("search API key not configured")
	}

	switch t.Provider {
	case "tavily":
		return t.searchTavily(context.Background(), args.Query)
	case "searxng":
		return t.searchSearxNG(context.Background(), args.Query)
	default:
		return "", fmt.Errorf("unknown search provider: %s", t.Provider)
	}
}

func (t *SearchTool) searchTavily(ctx context.Context, query string) (string, error) {
	body := map[string]interface{}{
		"api_key":      t.APIKey,
		"query":        query,
		"search_depth": "basic",
		"max_results":  5,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal tavily request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.tavily.com/search", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create tavily request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := searchHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("tavily request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read tavily response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("tavily API error %d: %s", resp.StatusCode, string(respBody))
	}

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

	if len(result.Results) == 0 {
		return "No results found.", nil
	}

	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	return sb.String(), nil
}

func (t *SearchTool) searchSearxNG(ctx context.Context, query string) (string, error) {
	baseURL := t.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	searchURL := fmt.Sprintf("%s/search?q=%s&format=json&categories=general",
		baseURL, url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return "", fmt.Errorf("create searxng request: %w", err)
	}

	resp, err := searchHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("searxng request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read searxng response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("searxng API error %d: %s", resp.StatusCode, string(respBody))
	}

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

	if len(result.Results) == 0 {
		return "No results found.", nil
	}

	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	return sb.String(), nil
}
