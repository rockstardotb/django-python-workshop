# django-python-workshop
This is a workshop utilizing python and docker-compose to familiarize oneself in architecture and containerization.

## The standard layout of an application (WITHOUT Docker) is as follows:


     app/
     │
     ├── requirements.txt
     │
     ├── setup.py
     │
     ├── README.md
     │
     └──app/
        │
        ├── __init__.py
        │
	├── database/
        │   │
	│   ├── __init__.py
        │   │
	│   └── db.sqlite3
        │
	├── main_cli.py
        │
	├── server/
        │   │
	│   ├── __init__.py
        │   │
	│   └── default.conf
        │
	└── utils/
            │
	    ├── __init__.py
            │
	    └── utils.py

## With Docker, we add segmentation for extra security by breaking the application into smaller pieces. 

 ___________                     _____________________                      _________
| database |===bridge network===| main_cli.py & utils |===bridge network===| server |=== ports 80, 443, etc

These containers are organized and built using the docker-compose command with a docker-compose.yml file. An example of a docker-compose.yml file that builds this sort of network is as follows:

    version: "3.2"
    services:
      djangoapp:
        build: .
        command: "gunicorn app.wsgi:application --bind 0.0.0.0:8000"
        networks:
          - nginx_network
          - db_network
        env_file:
          - ./.env
        restart: unless-stopped
        depends_on:
          - postgres
      postgres:
        image: postgres:11.0-alpine
        restart: unless-stopped
        volumes:
          - /database_directory:/var/lib/postgresql/data
        networks:
          - db_network
        env_file:
          - ./.env_db
      nginx:
        restart: unless-stopped
        build: 
            context: ./nginx/
        ports:
          - "80:80"
          - "443:443"
          - "587:587"
        volumes:
          - ./nginx/letsencrypt/:/etc/letsencrypt/
          - ./app/static/:/opt/services/djangoapp/static/
        depends_on:
          - djangoapp
        networks:
          - nginx_network
    networks:
      nginx_network:
        driver: bridge
      db_network:
        driver: bridge

#### Let us break it down:

The services represent the three components (or docker containers) that we broke the app down into. This is known as microsegmentation. Let's look at the 'djangoapp' service.

    djangoapp:
      build: .
      command: "gunicorn app.wsgi:application --bind 0.0.0.0:8000"
      networks:
        - nginx_network
        - db_network
      env_file:
        - ./.env
      restart: unless-stopped
      depends_on:
        - postgres

The 'build: .' tells docker-compose that the Dockerfile, which has all of the commands for dependency installations, is located.

The 'command: "gunicorn app.wsgi:application --bind 0.0.0.0:8000"' tells docker-compose to run the 'gunicorn' command when the container is run.

The 'networks:' section defines the bridge networks that are connected to the djangoapp container. In this case, we have one bridge network connecting the djangoapp to the database container and another, separate network connecting the web server container to the djangoapp container. Note, at the bottom of the entire docker-compose.yml file, not the subsection we are currently looking at, that we must define our networks and their drivers.

The 'env_file:' section is very important. For security reasons, we do not want to hardcode any sensitive information in our source code as it would be visible to potential hackers. Instead, we store those in a separate file called .env and docker-compose creates environmental variables from its contents. For example, in a python program, instead of setting psswd='12345', we'd write psswd=os.getenv('PSSWD'). Note, the '- ./.env' tells docker-compose that the .env file is located in the current directory.

The 'restart: unless-stopped' command tells docker-compose that, unless the container is explicitely stopped using the 'docker-compose down' command, that it should be restarted. An example scenario would be a reboot of the server. In the event of the reboot, the container would automatically start back up.

The 'depends_on:' command tells docker-compose to build and run other containers, in this case it is the container named 'postgres', before running the djangoapp container.

#### Other commands not used in the django app, but in one of the other containers are:

'volumes:' - mounts a connection between the two directories listed where the left side is on the host machine and the right side is a directory inside the container. Use this during development if you'd like for work done inside the container in that directory to persist after the container is stopped.

'ports:' - binds a port on the host (left side) to a port in the container (right side).

'context:' - used with the 'build' command if the Dockerfile to be used is located any where other than the current directory. Specifies the location of the Dockerfile.

#### Running your application

If you have not already built your container, or you've made changes to the underlying source code, build with

    docker-compose build

To run your container, run (add a ' -d' to the end if you want to run the application in the background)

    docker-compose up

To bring the application down, run

    docker-compose down

## Now, lets build an application that uses python and docker-compose. For simplicity, I am only going to use a single container, but I will still use docker-compose.yml.

The architecture will look similar to the following:

     app/
     │
     ├── requirements.txt
     │
     ├── docker-compose.yml
     │
     ├── Dockerfile
     │
     ├── README.md
     │
     └──app/
        │
        ├── __init__.py
        │
        │
        ├── main_cli.py
        │
        │
        └── utils/
            │
            ├── __init__.py
            │
            └── utils.py


I am going to write a machine learning application to make predictions using logistic regression. First, create a new directory for your app:

    mkdir app

Next, move into the new app directory.

    cd app

let us create the initial Dockerfile. I know I want to use python's machine learning stack, so I will choose a premade image from Docker, called 'waleedka/modern-deep-learning'.

    FROM waleedka/modern-deep-learning

next, I need to create my docker-compose.yml file:

    version: "3.2"
    services:
      djangoapp:
        build: .
        volumes:
          - ./app/:/app/
        restart: unless-stopped

Now I need to create the 'app/app/' directory

    mkdir app


