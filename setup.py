from distutils.core import setup

setup(name='app',
      version='1.0',
      description='tutorial on the use of docker and python',
      author='Alex Liddle',
      author_email='aliddle@rsitex.com',
      url='https://github.com/rockstardotb/docker-python-workshop',
      packages=['app',
                'app.utils',
                'app.data',
                ],
     )

