# Webshell

[TOC]

Detect webshell based on factorization machine

## 1 Data Preprocessing

### 1.1 Features

#### 1.1.1 Static Text Feature

#### 1.1.2 Other Feature

We use [NeoPI](https://github.com/Neohapsis/NeoPI) created by [Neohapsis](https://github.com/Neohapsis) to extract some important feature from our examples. 

```shell
./neopi.py --csv=neopiWebshell.csv --all /samples/webshell \.php
./neopi.py --csv=neopiNonwebshell.csv --all /samples/nonwebshell \.php
```







---

## What is webshell?

- Webshell是以asp、php、jsp或者cgi等网页文件形式存在的一种命令执行环境。黑客在入侵了一个网站后，通常会将webshell与网站服务器web目录下正常的网页文件混在一起，然后就可以使用浏览器来访问webshell，得到一个命令执行环境，以达到控制网站服务器的目的。

  [webshell]:https://baike.baidu.com/item/webshell/966625?fr=aladdin