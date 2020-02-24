FROM waleedka/modern-deep-learning

COPY ./ /app

WORKDIR /app

RUN pip3 install -e .

WORKDIR /app/app
