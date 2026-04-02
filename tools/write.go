package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type WriteTool struct{}

type WriteInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type EditInput struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func (t *WriteTool) Name() string {
	return "write"
}

func (t *WriteTool) Description() string {
	return "Create or overwrite a file with the given content. Use old_string and new_string for partial edits."
}

func (t *WriteTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "The content to write (for create/overwrite)",
			},
			"old_string": map[string]interface{}{
				"type":        "string",
				"description": "The text to replace (for partial edit)",
			},
			"new_string": map[string]interface{}{
				"type":        "string",
				"description": "The replacement text (for partial edit)",
			},
		},
		"required": []string{"path"},
	}
}

func (t *WriteTool) Execute(input json.RawMessage) (string, error) {
	var args map[string]interface{}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	// Partial edit mode
	if oldStr, ok := args["old_string"].(string); ok {
		newStr, _ := args["new_string"].(string)
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read %s for edit: %w", path, err)
		}
		content := string(data)
		if !strings.Contains(content, oldStr) {
			return "", fmt.Errorf("old_string not found in %s", path)
		}
		content = strings.Replace(content, oldStr, newStr, 1)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "", fmt.Errorf("write %s: %w", path, err)
		}
		return fmt.Sprintf("Successfully edited %s", path), nil
	}

	// Full write mode
	content, _ := args["content"].(string)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create directory %s: %w", dir, err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	return fmt.Sprintf("Successfully wrote %s", path), nil
}
