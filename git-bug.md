* githubͬ同一账号推送多个仓库:
    https://stackoverflow.com/questions/11656134/github-deploy-keys-how-do-i-authorize-more-than-one-repository-for-a-single-mac#

* git错误=>fatal: refusing to merge unrelated histories:
    http://cache.baiducontent.com/c?m=9f65cb4a8c8507ed19fa950d100b96204a05d93e788090412189d65f93130a1c187ba0fc7063565f8e993d7a00aa425deffb3c742a567bf18cc8fe0a8aefd56974d47223706ac01c05d36ff09c06709637902caef359b0e4a374c4f8c5d3a90e088b15583adba7ca4e45498a39ff416aa5b19939410c56e9b327648f4e765a882331f610ada773355bd4e1dd2d0a9e3dd0104ec0ef60e72912c454f85f4c7a17e61be71f51576abb0e61a2046653d3&p=8b2a9715d9c040a90db8cc604d5397&newp=c2769a47819c15b108e2977f0b40c1231610db2151d6d41f6b82c825d7331b001c3bbfb423251202d2c27e6502af4d59e0f1307033012ba3dda5c91d9fb4c57479df78606e&user=ba1idu&fm=sc&query=git+fatal%3A+refusing+to+merge+unrelated+histories&qid=9d6756000002be1a&p1=

* error: cannot spawn less: No such file or directory
    window上没有less指令，git默认使用less进行分页显示，执行下面这个命令，用cat来分页
    ```
    git config --global core.pager cat
    ```