This repo contains the code and data for [the paper](https://arxiv.org/abs/2106.02359) at the Findings of ACL 2021:

**How Good Is NLP? A Sober Look at NLP Tasks through the Lens of Social Impact** [[Paper link](https://arxiv.org/abs/2106.02359)]

by _Zhijing Jin, Geeticka Chauhan, Brian Tse, Mrinmaya Sachan, Rada Mihalcea_.

To cite the paper:

```bibtex
@inproceedings{jin2020good,
    title = "How Good Is {NLP}? {A} Sober Look at {NLP} Tasks through the Lens of Social Impact",
    author = "Jin, Zhijing  and
      Chauhan, Geeticka  and
      Tse, Brian  and
      Sachan, Mrinmaya  and
      Mihalcea, Rada",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2106.02359",
    doi = "10.18653/v1/2021.findings-acl.273",
    pages = "3099--3113",
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
