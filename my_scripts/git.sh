git config --local user.name "QMMMS"
git config --local user.email "1595168717@qq.com"
git config --list
git add .
git commit -m "update"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/qinmaosheng.id_rsa
git push