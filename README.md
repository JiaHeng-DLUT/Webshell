# Webshell

## 1 Instuction

1. Download and extract source code.
2. cd into the directory using the similar command as follows:

```shell
cd D:/webshell
```

3. Put the files to detect into the `samples2dectect` folder.
4. Run the similar command to get features.

```shell
./neopi.py -c neopiFeature.csv -a D:/webshell/samples2detect "php"
```

5. Run the following command to detect webshells.

```shell
python detect.py
```



