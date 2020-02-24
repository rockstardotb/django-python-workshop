# make build

cwd = $(shell pwd)
build:
	docker build -t mlearn .
# make run
run:
	docker run -i -t --name mlearn -v /${cwd}/:/app/ -d mlearn /bin/bash
# make exec
exec:
	docker exec -i -t mlearn /bin/bash
# start
start:
	docker start mlearn
# stop
stop:
	docker stop mlearn
# rm
remove:
	docker rm mlearn

