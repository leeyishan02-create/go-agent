package tools

import (
	"encoding/json"
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
		return "Error: invalid input: " + err.Error(), nil
	}

	path, _ := args["path"].(string)
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	// Partial edit mode
	if oldStr, ok := args["old_string"].(string); ok {
		newStr, _ := args["new_string"].(string)
		data, err := os.ReadFile(path)
		if err != nil {
			return "Error reading file for edit: " + err.Error(), nil
		}
		content := string(data)
		if !strings.Contains(content, oldStr) {
			return "Error: old_string not found in file", nil
		}
		content = strings.Replace(content, oldStr, newStr, 1)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "Error writing file: " + err.Error(), nil
		}
		return "Successfully edited " + path, nil
	}

	// Full write mode
	content, _ := args["content"].(string)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "Error creating directory: " + err.Error(), nil
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "Error writing file: " + err.Error(), nil
	}
	return "Successfully wrote " + path, nil
}
