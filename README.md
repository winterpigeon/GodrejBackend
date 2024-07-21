# GodrejBackend

python3 -m uvicorn main:app

source activate pytorch

nohup uvicorn main:app &

tail -f nohup.out

ps aux | grep uvicorn

kill -9 pid
