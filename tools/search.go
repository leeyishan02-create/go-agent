package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

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
		return "Error: invalid input: " + err.Error(), nil
	}

	if t.APIKey == "" {
		return "Error: search API key not configured. Please set it in Settings.", nil
	}

	switch t.Provider {
	case "tavily":
		return t.searchTavily(args.Query)
	case "searxng":
		return t.searchSearxNG(args.Query)
	default:
		return "Error: unknown search provider: " + t.Provider, nil
	}
}

func (t *SearchTool) searchTavily(query string) (string, error) {
	body := map[string]interface{}{
		"api_key":      t.APIKey,
		"query":        query,
		"search_depth": "basic",
		"max_results":  5,
	}
	bodyBytes, _ := json.Marshal(body)

	resp, err := http.Post("https://api.tavily.com/search", "application/json", strings.NewReader(string(bodyBytes)))
	if err != nil {
		return "Search error: " + err.Error(), nil
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "Search API error: " + string(respBody), nil
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
		return "Search parse error: " + err.Error(), nil
	}

	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	if sb.Len() == 0 {
		return "No results found.", nil
	}
	return sb.String(), nil
}

func (t *SearchTool) searchSearxNG(query string) (string, error) {
	baseURL := t.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	searchURL := fmt.Sprintf("%s/search?q=%s&format=json&categories=general",
		baseURL, url.QueryEscape(query))

	resp, err := http.Get(searchURL)
	if err != nil {
		return "Search error: " + err.Error(), nil
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "Search API error: " + string(respBody), nil
	}

	var result struct {
		Results []struct {
			Title    string `json:"title"`
			URL      string `json:"url"`
			Content  string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "Search parse error: " + err.Error(), nil
	}

	var sb strings.Builder
	for i, r := range result.Results {
		sb.WriteString(fmt.Sprintf("[%d] %s\n    URL: %s\n    %s\n\n", i+1, r.Title, r.URL, r.Content))
	}
	if sb.Len() == 0 {
		return "No results found.", nil
	}
	return sb.String(), nil
}
