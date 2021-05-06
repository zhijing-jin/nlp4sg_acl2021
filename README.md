This repo contains the code and data for the paper at ACL Findings 2021:

**How Good Is NLP? A Sober Look at NLP Tasks through the Lens of Social Impact**

by _Zhijing Jin, Geeticka Chauhan, Brian Tse, Mrinmaya Sachan, Rada Mihalcea_.

To cite the paper:

```bibtex
@inproceedings{jin2020good,
    title = {How Good Is NLP? A Sober Look at NLP Tasks through the Lens of Social Impact},
    author = {Zhijing Jin and Geeticka Chauhan and Brian Tse and Mrinmaya Sachan and Rada Mihalcea},
    booktitle = {Findings of ACL},
    year = {2021},
}
```

### Visualizations

To check out the visualizations of NLP4SocialGood papers, please see the image files
in [`data/visualization/`](data/visualization)

### Annotated Data

See [`data/acl_long_clean.csv`](data/acl_long_clean.csv) for our final data.

The format is (paper_name, Stage, track, social good domain, author)

- `paper_name`: taken from the list
  of [accepted papers](https://www.aclweb.org/anthology/events/acl-2020/#2020-acl-main) at ACL 2020
- `Stage`: taking values from {1,2,3,4}, classified according to Section 3.1 of our position paper
- `track`: adapted from the [ACL tracks](https://acl2020.org/blog/general-conference-statistics/)
- `social good domain`: taking values from {bias mitigation, education, equality, fighting misinformation, green NLP,
  healthcare, interpretability, legal applications, low-resource language, mental healthcare, robustness, science
  literature parsing, and others}
- `author`: taken from the list of [accepted papers](https://www.aclweb.org/anthology/events/acl-2020/#2020-acl-main) at
  ACL 2020

### Codes

#### Preparing for the environment

We use Python 3.

```bash
pip install code/requirements.txt
```

#### Run the code

```bash
python code/visualize_data.py
```

### Pull Requests

We welcome [Pull Requests](https://github.com/zhijing-jin/nlp4sg_acl2021/pulls) on improving the data csv files, or the
codes.