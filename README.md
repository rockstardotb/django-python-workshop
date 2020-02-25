# docker-python-workshop
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

The 'build: .' tells docker-compose where the Dockerfile, which has all of the commands for dependency installations, is located.

The 'command: "gunicorn app.wsgi:application --bind 0.0.0.0:8000"' tells docker-compose to run the 'gunicorn' command when the container is run.

The 'networks:' section defines the bridge networks that are connected to the djangoapp container. In this case, we have one bridge network connecting the djangoapp to the database container and another, separate network connecting the web server container to the djangoapp container. Note, at the bottom of the entire docker-compose.yml file, not the subsection we are currently looking at, that we must define our networks and their drivers.

The 'env_file:' section is very important. For security reasons, we do not want to hardcode any sensitive information in our source code as it would be visible to potential hackers. Instead, we store those in a separate file called .env and docker-compose creates environmental variables from its contents. For example, in a python program, instead of setting psswd='12345', we'd write psswd=os.getenv('PSSWD'). Note, the '- ./.env' tells docker-compose that the .env file is located in the current directory.

The 'restart: unless-stopped' command tells docker-compose that, unless the container is explicitely stopped using the 'docker-compose down' command, that it should be restarted. An example scenario would be a reboot of the server. In the event of the reboot, the container would automatically start back up.

The 'depends_on:' command tells docker-compose to build and run other containers, in this case it is the container named 'postgres', before running the djangoapp container.

#### Other commands not used in the django app, but in one of the other containers are:

'volumes:' - mounts a connection between the two directories listed where the left side is on the host machine and the right side is a directory inside the container. Use this during development if you'd like for work done inside the container in that directory to persist after the container is stopped.

'ports:' - binds a port on the host (left side) to a port in the container (right side).

'context:' - used with the 'build' command if the Dockerfile to be used is located any where other than the current directory. Specifies the location of the Dockerfile.

#### Running your application (Note, we haven't made our Dockerfile yet, so this will fail. This is purely informational at this point in the workshop)

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
    
    COPY ./ /app
    
    WORKDIR /app

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

For now, because we don't have any programs written yet, the docker container is going to build, run, and then stop as there are no tasks. So, for now, we are going to run the container without compose until we have written our application. The commands for building and running containers can get pretty lengthy. For that reason, I like to create a Makefile to shorten my commands. In your current directory create a Makefile and insert the following:

    # make build

    cwd = $(shell pwd)
    build:
    	docker build -t mlearn .
    # make run
    run:
    	docker run -i -t --name mlearn -v /${cwd}:/app/ -d mlearn /bin/bash
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
    
Let us build, run, and exec into our container

    make build
    make run
    make exec

Now, the first thing we need to do is create our '__init__.py'. This tells python that any modules in the current directory can be imported into any other modules.

    touch /app/app/__init__.py


Let us also make our utils directory and move into it.

    mkdir /app/app/utils
    cd /app/app/utils/

The specific algorithm that I want to build involves several steps. First, data must be imported, cleaned, and transformed. Within the app/app/utils/ directory, create a file called 'preprocessing.py' and insert the following into it:

    # Data Preprocessing
    
    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Importing the dataset
    def import_data(csv, xidx, yidx, start=0, header=0):
        dataset = ""
        if header is None:
            dataset = pd.read_csv(csv,header=header)
        else:
            dataset = pd.read_csv(csv)
        X = dataset.iloc[:, start:xidx].values
        y = dataset.iloc[:, yidx].values
        return X,y
    
    # Taking care of missing data
    def fix_missing(X, xstart, xstop):
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(X[:, xstart:xstop])
        X[:, xstart:xstop] = imputer.transform(X[:, xstart:xstop])
        return X
    
    # Encoding categorical data
    def categorical_encode(data, independent=True, equal=True, idx=0):
        # Encoding the Independent Variable
        # [string1,string2,string3] --> [0,1,2]
        if independent:
            labelencoder_data = LabelEncoder()
            data[:, idx] = labelencoder_data.fit_transform(data[:, idx])
    
            if equal:
                from sklearn.preprocessing import OneHotEncoder
                # Prevent machine from thinking one category is greater than
                # another
                # [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
                # first column --> France, second column --> Germany, third column --> Spain
                onehotencoder = OneHotEncoder(categorical_features = [idx])
                data = onehotencoder.fit_transform(data).toarray()
    
        else:
            # Encoding the Dependent Variable
            # Dependent variable doesn't need OneHotEncoder
            # ['No','Yes'] --> [0,1]
            labelencoder_data = LabelEncoder()
            data = labelencoder_data.fit_transform(data)
    
        return data
    
    # Split dataset into training and test sets
    def create_sets(X, y, size=0.2, random_state=0):
    
            return train_test_split(X, y, test_size = size, random_state = random_state)
    
    # Feature scaling
    def feature_scale(X_train, X_test=None):
            # Put columns in same scale so one feature doesn't
            # dominate another
    
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            if X_test is None:
                return X_train, sc_X
    
            X_test = sc_X.transform(X_test)
    
            return X_train, X_test, sc_X

Note, this workshop is not intended to teach machine learning, but, instead, it is intended to teach about software architecture, syntax, etc. That being said, lets analyze the syntax and formation of one of the functions in preprocessing.py:

    # Encoding categorical data
    def categorical_encode(data, independent=True, equal=True, idx=0):
        # Encoding the Independent Variable
        # [string1,string2,string3] --> [0,1,2]
        if independent:
            labelencoder_data = LabelEncoder()
            data[:, idx] = labelencoder_data.fit_transform(data[:, idx])
    
            if equal:
                # Prevent machine from thinking one category is greater than
                # another
                # [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
                # first column --> France, second column --> Germany, third column --> Spain
                onehotencoder = OneHotEncoder(categorical_features = [idx])
                data = onehotencoder.fit_transform(data).toarray()
    
        else:
            # Encoding the Dependent Variable
            # Dependent variable doesn't need OneHotEncoder
            # ['No','Yes'] --> [0,1]
            labelencoder_data = LabelEncoder()
            data = labelencoder_data.fit_transform(data)
    
        return data

Syntax: standard syntax for functions is lower-case and underscore between words (i.e. categorical_encode). Variables follow the same rules. The naming convention for classes is camel-case (i.e., OneHotEncoder). 
Inline Documentation: Besides using self-documenting code, i.e., naming things in a way so that it is obvious what each part of the code is doing, we also use inline comments utilizing the #-sign.
Initializing Parameters: The parameters of the function are initialized with default values. This is useful for mitigating missing parameters resulting from user input, or saving time and space when the majority of instances use the default values.

Now, let us create another file in the utils directory called logistic_regression.py and insert the following into it.

    from sklearn.linear_model import LogisticRegression as logreg
    
    # fit data to logistic curve
    def train(X_train,y_train,random_state=0):
        regressor = logreg(random_state=random_state)
        regressor.fit(X_train,y_train)
        return regressor
    
    # Make a prediction using the trained model
    def predict(X_test, regressor):
        y_pred = regressor.predict(X_test)
    
        return y_pred
    
    # Plot expected vs predicted
    def visualize(X_train, y_train, regressor):
        import matplotlib.pyplot as plt
        import numpy as np
    
        X_grid = np.arange(min(X_train),max(X_train),0.001)
        X_grid = X_grid.reshape((len(X_grid),1))
        plt.scatter(X_train, y_train, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
        plt.title('Y vs X (Training Set)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    # Create a confusion matrix [[correct,incorrect],[incorrect,correct]]
    def confusion(X_test,y_test, X_train, y_train, regressor):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np
        from sklearn.metrics import confusion_matrix as cmat
    
        y_pred = predict(X_test,regressor)
        cm = cmat(y_test,y_pred)
    
        X_set, y_set = X_train,y_train
    
        X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                             np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
        plt.contour(X1,X2,regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
        plt.xlim(X1.min(),X1.max())
        plt.ylim(X2.min(),X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                        c=ListedColormap(('red','green'))(i), label=j)
        plt.title('Logistic Regression (Training set)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    
        return cm

Note, this is not our main module, but it contains a list of functions that we will use, hence it belongs in the utils directory. If, perhaps, we had multiple regression modules, I would separate them into their own directory called 'regression', but, because there is just one module, putting it inside the utils directory is fine.

Next, we need some data. Create a data/ directory in app/app/, download <a href="./data/Social_Media_Ads.csv" download>Social_Media_Ads.csv</a>, and put it in the /app/app/data directory:

    mkdir /app/app/data

Now your app structure should look like this:

     app/
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
        ├── data/
        │   │
        │   └── Social_Media_Ads.csv
        │
        │
        └── utils/
            │
            ├── __init__.py
            │
            ├── logistic_regression.py
            │
            └── preprocessing.py

Now it is time for our main client file. In /app/app/, create main_cli.py and insert the following into it:

    from app.utils import preprocessing as prep
    from app.utils import logistic_regression as logreg
    
    class TestLogreg(object):
    
        # Initialize your class attributes
        def __init__(self, data_path="", X=[], y=[], split=0.25, X_train=[], y_train=[],
                     X_test=[], y_test=[], sc_X=[]):
    
            self.data = data_path
            self.X = X
            self.y = y
            self.split = split
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.sc_X = sc_X
    
            self.prep_data()
            self.regressor = self.train()
            self.result = self.predict()
    
        def prep_data(self):
            self.X, self.y = prep.import_data(self.data, 4, 4, 2)
            self.X = self.X.astype(float)
            self.X_train, self.X_test, self.y_train, self.y_test = prep.create_sets(self.X, self.y, size=self.split)
            self.X_train, self.X_test, self.sc_X = prep.feature_scale(self.X_train, self.X_test)
    
        def train(self):
            return logreg.train(self.X_train,self.y_train,random_state=0)
    
        def predict(self):
            return self.regressor.predict(self.X_test)
    
        def accuracy(self):
            correct = 0
            for i in range(len(self.result)):
                if self.result[i] == self.y_test[i]:
                    correct += 1
    
            return correct*100/len(self.X_test)
    
    if __name__ == '__main__':
    
        test = Test_logreg(data_path='./data/Social_Network_Ads.csv')
    
        print('\nModel: Logistic Regression\nAccuracy: {}%\n'.format(test.accuracy()))

Lets break down the Class structure:

Note, we use camel-case for the title of the class and we define it as an object type:

    class TestLogreg(object):

The first function we define is the __init__() function, in which we set default values for each parameter:

        def __init__(self, data_path="", X=[], y=[], split=0.25, X_train=[], y_train=[],
                     X_test=[], y_test=[], sc_X=[]):

            self.data = data_path
            self.X = X
            self.y = y
            self.split = split
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.sc_X = sc_X

            self.prep_data()
            self.regressor = self.train()
            self.result = self.predict()

All other functions (or methods) follow the __init__() method.

I write the main function after class, in which I initialize and instance of the class and call the accuracy method:

    if __name__ == '__main__':

        test = Test_logreg(data_path='./data/Social_Network_Ads.csv')

        print('\nModel: Logistic Regression\nAccuracy: {}%\n'.format(test.accuracy()))

Finally, to import our custom modules, we need to write an /app/setup.py file consisting of the following:

    from distutils.core import setup

    setup(name='app',
          version='1.0',
          description='tutorial on the use of docker and python',
          author='Alex Liddle',
          author_email='alex37184@yahoo.com',
          url='https://github.com/rockstardotb/docker-python-workshop',
          packages=['app',
                    'app.utils',
                    'app.data',
                    ],
         )

and we need to add the following to the bottom of our /app/Dockerfile:

    RUN pip3 install -e .
    
    WORKDIR /app/app

Now your app structure should look like this:

     app/
     │
     ├── docker-compose.yml
     │
     ├── Dockerfile
     │
     ├── README.md
     │
     ├── setup.py
     │
     └──app/
        │
        ├── __init__.py
        │
        │
        ├── data/
        │   │
        │   └── Social_Media_Ads.csv
        │
        │
        └── utils/
            │
            ├── __init__.py
            │
            ├── logistic_regression.py
            │
            └── preprocessing.py

Now lets exit, rebuild and run our docker container, and then exec in to run a test:

    exit
    make stop
    make remove
    make build
    make run
    make exec

    python main_cli.py

If the output of 'python main_cli.py' is...

    Model: Logistic Regression
    Accuracy: 89.0%

...then you have correctly built the application.

Now we can automate the build, exec, and run process with docker-compose, first just change the content
of docker-compose.yml to:

    version: "3.2"
    services:
            djangoapp:
                    build:
                            context: ${PWD}
                    command: python3 main_cli.py
    

Now run docker-compose build && docker-compose up

again, if the output is...

    Model: Logistic Regression
    Accuracy: 89.0%

...then you have correctly built the application.

Congratulations, you have now used docker-compose with python. Remember, though this application seems simple and the use of docker-compose over docker seems unecessary, you'll appreciate docker-compose when building full-stack applications containing a database, app, and webserver. docker-compose is useful for service orchestration and microsegmentation.
