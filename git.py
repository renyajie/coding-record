a brief records about some useful operations in git.

*init config:
    1. git config --global user.name 'username' : 在用户主目录下配置用户名
    2. git config --global user.email 'email' : 在用户主目录下配置邮箱
    3. --global 表示全局配置，我们也可以在某个项目的目录下单独配置，去掉这个参数即可

*base operation:
    1. git init: 让当前文件夹变成git可以管理的仓库
    2. git add xxx: 将xxx添加到仓库的暂存区
    3. git commit -m xxx: 将暂存区的所有文件提交到仓库，-m 参数用来指定提交的信息
    4. git status: 查看仓库当前状态
    5. git diff xxx: 查看对xxx文件进行了哪些修改:
        5.1 git diff HEAD -- xxx: 查看工作区的xxx文件和版本库中最新版本的区别
    6. git log: 查看提交历史，可能会删减:
        6.1 git log --pretty=oneline: 查看简略版的提交历史
        6.2 git reflog: 查看每一次的提交历史，没有删减
    7. git reset: 回退命令:
        7.1 git reset --hard HEAD^: 回退到上一个版本
        7.2 git reset --hard HEAD~n: 回退到上n个版本
        7.3 git reset --hard xxxxxx: 回退到commitId是xxxxxx的版本
        7.4 giy reset HEAD xxx: 将暂存区的修改撤销掉，重新放回工作区
    8. git checkout -- xxx: 将工作区的xxx修改(添加修改,甚至是删除)回退到最近commit(未添加进暂存区)或add(已添加进暂存区)状态
    9. git rm xxx: 从仓库中删除xxx文件
	10. git rebase: 将本地未push的操作整理成一条直线

*remote repository:
    1. git remote add origin git@github.com:xxx/aaa.git:
        将此时文件夹的项目与github上xxx作者的aaa项目建立关联
    2. git push -u origin master: 将本地的master分支推送到远端仓库，并建立链接:
        2.1 git push origin dev: 将本地的dev分支推送到远端
    3. git clone git@github.com:xxx/aaa.git: 将xxx作者的aaa项目克隆到本地
    4. git remote: 查看远程库信息:
        4.1 git remote -v: 查看远程库信息，详细版
    5. git pull: 将远程库的当前分支的最新提交抓取下来
    6. git branch --set-upstream dev origin/dev: 将本地dev分支与远程库dev分支进行关联

*branch management:
    1. git checkout -b dev: 创建dev分支并切换到dev:
        1.1 git checkout -b dev origin/dev: 创建分支dev并与远程仓库进行关联
    2. git merge dev: 将dev分支合并到当前分支，不保留合并信息:
        2.1 git merge --no-ff -m 'xxxx' dev: 警用fast forward合并，保留合并记录，并以xxx作为提交信息
    3. git branch -d dev: 删除dev分支:
        3.1 git branch: 查看dev分支
        3.2 git branch xxx: 创建xxx分支，不切换
        3.3 git branch -D dev: 在未合并的情况下，强行删除dev分支
    4. git log --graph --pretty=oneline --abbrev-commit: 查看分支合并情况:
        4.1 git log --graph --pretty=oneline: 查看分支合并图，详细一些
        4.2 git log --graph: 查看分支合并图，无删减版
    5. git stash:
        将当前工作区间储存起来，等回复现场后继续工作，通常用于开发中途中断的情况，或要立即切换分支工作:
        5.1 git stash list: 查看工作保存记录
        5.2 git stash pop: 回复工作现场，并删除stash中的工作记录
        5.3 git stash apply stash@{0}: 恢复stash@{0}的工作现场

*tag management:
    1. git tag v1.0: 在当前分支打上v1.0的标签:
        1.1 git tag: 查看所有标签
        1.2 git tag -a v0.1 -m 'xxx' abcdef:
            给commitId是abcdef的提交打上v0.1的标签，并且提交信息为xxx
        1.3 git tag -d v0.1: 将本地的v0.1标签删除
        1.4 git push origin v1.0: 将v1.0标签推送到远端
        1.5 git push origin --tags: 将未推送的本地标签一次性全部推送到远端
        1.6 git push origin :refs/tags/v0.1: 在1.3后执行，可删除远端v0.1的标签

    2. git show v0.1: 查看v0.1的标签信息

*git customize:
    1 .gitignore: 在当前项目文件下创建.gitignore，git会忽略其中定义的文件:
        1.1 git add -f xxx.aa: 将被.gitignore忽略的aa形式文件强行加入仓库
    2. git config --global alias.st status: 定义status的别名st，可简化命令，其他可以
