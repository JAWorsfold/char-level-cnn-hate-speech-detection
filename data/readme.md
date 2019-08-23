### Data sets

These have been separated into **public** and **private** directories due to twitter's policy not to share tweet 
statuses online. Only the public directory has been pushed to this repository.

#### public
This contains labeled data which only contains the tweet ID and not the tweet status.
The Twitter API was utilised in order to access the tweet statuses. The code used to do so can be found in
`twitter_api.py`. For more information on the data therein please open the directory.

#### private
This contains all of the data sets which include tweet statuses. The majority of the data was sourced by Davidson T., 
et al. and is accessible [here.](https://github.com/t-davidson/hate-speech-and-offensive-language)
If you wish to use this data set please cite the paper by Davison, T. et al.:

~~~
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
~~~

This directory also contains the combined public and private data sets with tweet statuses, as well as any preprocessed
versions of the data.
