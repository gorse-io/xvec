package snowball

import (
	"bytes"
	"log"
	"testing"
)

func TestEnvDebugProducesNoOutput(t *testing.T) {
	var output bytes.Buffer
	originalWriter := log.Writer()
	log.SetOutput(&output)
	t.Cleanup(func() {
		log.SetOutput(originalWriter)
	})

	NewEnv("test").Debug(1, 2)

	if output.Len() != 0 {
		t.Fatalf("Debug wrote unexpected output: %q", output.String())
	}
}
