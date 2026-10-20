package host

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
)

// placedProblem says in plain words why a placed model file failed its
// proof, with what fixes it (Decision 2, conditions 5 and 6), or returns ""
// for any other failure. download is how the person gets the file again.
func placedProblem(err error, download string) string {
	var unreadable *programfiles.UnreadableError
	if errors.As(err, &unreadable) && stillBeingWritten(err) {
		return fmt.Sprintf("the model file %s is still being written by another program; try again once it finishes", filepath.FromSlash(unreadable.Path))
	}
	var mismatch *programfiles.MismatchError
	if errors.As(err, &mismatch) && mismatch.Kind == programfiles.Changed {
		return fmt.Sprintf("the model file %s no longer matches its download; %s", filepath.FromSlash(mismatch.Path), download)
	}
	return ""
}

// Repair paths for a placed model file that no longer matches.
const (
	downloadVideoAgain = "download it again from the archive or the internet, then place it with modelhostd models install --kind sdcpp"
	downloadChatAgain  = "download the model again from the archive or the internet"
)
