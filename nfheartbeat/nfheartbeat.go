// Package nfheartbeat sends the periodic NF heartbeat of 3GPP TS 29.510 clause 5.2.2.3.2.
package nfheartbeat

import (
	"context"
	"errors"
	"net/http"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/free5gc/openapi"
	"github.com/free5gc/openapi/models"
)

const (
	// DefaultTimer is the fallback interval in seconds. Equal to the free5gc NRF
	// enforcement default: longer would cross the deadline where it suspends a silent NF.
	DefaultTimer int32 = 10

	// MinTimer and MaxTimer mirror the free5gc NRF heartBeatTimer validation bounds.
	MinTimer int32 = 1
	MaxTimer int32 = 3600
)

type hbStatus int

const (
	hbStatusOk hbStatus = iota
	hbStatusNotFound
	hbStatusFailed
)

// Covers NRF database loss behind OAuth2: the token request fails before a PATCH can see the 404.
const reregisterFailureThreshold = 3

// PatchItems returns the NF heartbeat body from 3GPP TS 29.510 clause 5.2.2.3.2.
func PatchItems() []models.PatchItem {
	return []models.PatchItem{{
		Op:    models.PatchOperation_REPLACE,
		Path:  "/nfStatus",
		Value: models.Nrf_NFMgmt_NFStatus_REGISTERED,
	}}
}

// Registrar executes the NRF requests the Runner decides to send. Each NF implements it
// over its own consumer, so the SBI clients, OAuth2 and the NF profile stay NF-side.
type Registrar interface {
	// A non-2xx must come back as a non-nil ProblemDetails or an error carrying
	// openapi.GenericOpenAPIError, so the Runner can classify a 404. ctx is cancelled
	// on shutdown only: bound the request below the heartbeat interval.
	UpdateNFInstance(ctx context.Context, patchItems []models.PatchItem) (
		models.Nrf_NFMgmt_NFProfile, *models.ProblemDetails, error)
	// Returns the heartBeatTimer the NRF assigned, in seconds, 0 for none. May retry
	// internally until it succeeds or ctx is done.
	RegisterNFInstance(ctx context.Context) (int32, error)
}

// Runner sends the periodic NF heartbeat and re-registers the profile when the NRF has
// lost it. Only the first Start call has any effect.
type Runner struct {
	registrar     Registrar
	fallbackTimer func() int32
	log           *logrus.Entry

	started atomic.Bool
	wg      sync.WaitGroup

	// Seeded by Start, then owned by the heartbeat goroutine, so no lock. timer is in seconds.
	timer    int32
	failures int
}

// NewRunner returns a Runner driving registrar. fallbackTimer supplies the interval until
// the NRF assigns one; nil or non-positive means DefaultTimer.
func NewRunner(registrar Registrar, fallbackTimer func() int32, log *logrus.Entry) (*Runner, error) {
	if registrar == nil {
		return nil, errors.New("registrar cannot be nil")
	}
	if log == nil {
		return nil, errors.New("log cannot be nil")
	}
	return &Runner{
		registrar:     registrar,
		fallbackTimer: fallbackTimer,
		log:           log,
	}, nil
}

// Start launches the heartbeat; call it after a successful registration. nrfTimer is the
// heartBeatTimer assigned there, in seconds, 0 for none.
func (r *Runner) Start(ctx context.Context, wg *sync.WaitGroup, nrfTimer int32) {
	// A Start that raced shutdown must not begin heartbeating.
	if ctx.Err() != nil {
		return
	}
	if !r.started.CompareAndSwap(false, true) {
		r.log.Warnln("NF heartbeat already started, ignoring Start")
		return
	}
	r.timer = r.capTimer(nrfTimer)
	// The external Add comes first: a nil wg then panics before the private WaitGroup is
	// touched, so Wait cannot block on a goroutine never started.
	wg.Add(1)
	r.wg.Add(1)
	go func() {
		defer wg.Done()
		defer r.wg.Done()
		r.loop(ctx)
	}()
}

// Wait blocks until the heartbeat goroutine has exited, so nothing reaches the NRF after
// deregistration.
func (r *Runner) Wait() {
	r.wg.Wait()
}

func (r *Runner) loop(ctx context.Context) {
	// A panic escaping this goroutine would crash the process; the heartbeat then stays down.
	defer func() {
		if p := recover(); p != nil {
			r.log.Errorf("panic: %v\n%s", p, string(debug.Stack()))
		}
	}()

	interval := r.interval()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	r.log.Infof("NF heartbeat started, interval %v", interval)

	for {
		select {
		case <-ctx.Done():
			r.log.Infoln("NF heartbeat stopped")
			return
		case <-ticker.C:
			r.tick(ctx)
			if next := r.interval(); next != interval {
				interval = next
				ticker.Reset(interval)
				r.log.Infof("NF heartbeat interval updated to %v", interval)
			}
		}
	}
}

func (r *Runner) tick(ctx context.Context) {
	// Contains a panic from the re-registration path; guardedOnce covers the heartbeat.
	defer func() {
		if p := recover(); p != nil {
			r.log.Errorf("panic during NF re-registration: %v\n%s", p, string(debug.Stack()))
		}
	}()

	// A tick that fired alongside the shutdown must not reach the NRF.
	if ctx.Err() != nil {
		return
	}

	switch r.guardedOnce(ctx) {
	case hbStatusOk:
		r.failures = 0
		return
	case hbStatusNotFound:
		r.log.Warnln("NF profile not found on NRF, re-registering")
	case hbStatusFailed:
		r.failures++
		if r.failures < reregisterFailureThreshold {
			return
		}
		r.log.Warnf("%d consecutive NF heartbeat failures, re-registering", r.failures)
	}
	if r.recoverRegistration(ctx) {
		r.failures = 0
	}
}

// guardedOnce turns a panic into one failed heartbeat instead of the end of all of them.
func (r *Runner) guardedOnce(ctx context.Context) (status hbStatus) {
	defer func() {
		if p := recover(); p != nil {
			r.log.Errorf("panic during NF heartbeat: %v\n%s", p, string(debug.Stack()))
			status = hbStatusFailed
		}
	}()
	return r.once(ctx)
}

// A heartBeatTimer of 0 keeps the current interval: the int32 model cannot tell an explicit
// 0 from an absent field, so it must not read as disable.
func (r *Runner) once(ctx context.Context) hbStatus {
	nf, problemDetails, err := r.registrar.UpdateNFInstance(ctx, PatchItems())
	if err == nil && problemDetails == nil {
		if nf.HeartBeatTimer > 0 {
			r.timer = r.capTimer(nf.HeartBeatTimer)
		}
		return hbStatusOk
	}
	if isNotFound(problemDetails, err) {
		return hbStatusNotFound
	}
	r.log.Warnf("NF heartbeat failed: pd=%+v err=%+v", problemDetails, err)
	return hbStatusFailed
}

// isNotFound reports whether the NRF answered 404, in any shape a Registrar delivers it.
func isNotFound(pd *models.ProblemDetails, err error) bool {
	if pd != nil && pd.Status == http.StatusNotFound {
		return true
	}
	var apiErr openapi.GenericOpenAPIError
	if errors.As(err, &apiErr) && apiErr.ErrorStatus == http.StatusNotFound {
		return true
	}
	var apiErrPtr *openapi.GenericOpenAPIError
	if errors.As(err, &apiErrPtr) && apiErrPtr.ErrorStatus == http.StatusNotFound {
		return true
	}
	return false
}

// Gives up during shutdown: a PUT after deregistration would resurrect the profile.
func (r *Runner) recoverRegistration(ctx context.Context) bool {
	if ctx.Err() != nil {
		return false
	}
	timer, err := r.registrar.RegisterNFInstance(ctx)
	if err != nil {
		r.log.Errorf("NF re-registration aborted: %+v", err)
		return false
	}
	r.timer = r.capTimer(timer)
	r.log.Infoln("NF re-registered to NRF")
	return true
}

// capTimer caps at MaxTimer so a misbehaving NRF cannot park the heartbeat for hours.
// Non-positive values keep their fallback meaning, so MinTimer needs no clamp.
func (r *Runner) capTimer(timer int32) int32 {
	if timer > MaxTimer {
		r.log.Warnf("NRF-assigned heartbeat timer %ds capped to %ds", timer, MaxTimer)
		return MaxTimer
	}
	return timer
}

// Never non-positive: the ticker panics on zero.
func (r *Runner) interval() time.Duration {
	timer := r.timer
	if timer <= 0 && r.fallbackTimer != nil {
		timer = r.fallbackTimer()
	}
	if timer <= 0 {
		timer = DefaultTimer
	}
	return time.Duration(timer) * time.Second
}
