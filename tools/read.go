package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type ReadTool struct{}

type ReadInput struct {
	Path string `json:"path"`
}

func (t *ReadTool) Name() string {
	return "read"
}

func (t *ReadTool) Description() string {
	return "Read the contents of a file. Supports text files, and returns an error for binary files."
}

func (t *ReadTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file to read",
			},
		},
		"required": []string{"path"},
	}
}

func (t *ReadTool) Execute(input json.RawMessage) (string, error) {
	var args ReadInput
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	if args.Path == "" {
		return "", fmt.Errorf("path is required")
	}

	path := args.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("access %s: %w", path, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory, not a file", path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	content := string(data)
	const maxLen = 50000
	if len(content) > maxLen {
		content = content[:maxLen] + fmt.Sprintf("\n\n... (truncated, file is %d chars)", len(content))
	}

	return content, nil
}
