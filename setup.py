from setuptools import setup, find_packages

setup(
      name='bce',
      version='1.0.0',
<<<<<<< HEAD
      description='Bayesian Ensemble Clustering based on the paper: Bayesian Cluster Ensembles, 2009',
=======
      description='Bayesian Ensemble Clustering based on the paper Bayesian Cluster Ensembles, 2009',
>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f
      url='https://github.com/frUTCS16/clust',
      author='Farzan Memarian',
      author_email='farzan.memarian@utexas.edu',
      license='LICENSE.txt',
        
      classifiers=[
          'Developint Status :: 1 - Beta',
          'Programming Language :: Python :: 3',
          ],
      keywords='ensemble bayesian clustering',
      packages=find_packages(), # remove the exclude if you want to distribute the source
      install_requires=['requests'],
      package_data={

      },
      # package_data={'sample': ['package_data.dat']},
      zip_safe=False
      # data_files=[('Iris.mat', ['data/Iris.mat'])]
      )

