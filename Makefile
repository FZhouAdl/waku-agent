# launch-jarvis — one command per pillar.

.PHONY: run voice telegram trace eval eval-judge gate lint

run:            ## chat with Jarvis in the terminal
	python -m jarvis

voice:          ## talk to it — push-to-talk (pip install -e '.[voice]')
	python -m jarvis voice

telegram:       ## phone → laptop (needs TELEGRAM_BOT_TOKEN in .env)
	python -m jarvis telegram

trace:          ## local trace dashboard at http://localhost:6006
	phoenix serve

eval:           ## deterministic evals (0/1, no judge involved)
	python -m pytest -q evals/deterministic

eval-judge:     ## LLM-as-judge evals (scored %, needs ANTHROPIC_API_KEY)
	python -m pytest -q evals/judge

gate:           ## the release gate: deterministic must pass, judge must clear threshold
	python -m jarvis.ops.release_gate

lint:
	ruff check jarvis evals
