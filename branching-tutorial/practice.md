# Author: Gideon Kimaiyo
# June 21, 2023

### Before you begin, please configure your git name and email
[https://confluence.atlassian.com/bitbucket/configure-your-dvcs-username-for-commits-950301867.html]

### Instructions
- Create a new directory/folder and name it **git-branching**
- Open the terminal or git bash if on windows and navigate to the root directory of this project which is **git-branching**
- Check if you have git installed by typing on the terminal/git bash
```
git version
```
- If git version number is returned, then you are good to the next step, else follow this link [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install git on your PC
- Initialize a Git repository: 
```
git init -b master
```
- The command above initializes a Git repository and creates a git branch called **master**
- Create a new file in the root directory and name it **hello.txt**
- Add the sententce **hello world** to the text file and save
- See if there are any changes that need to be committed
```
git status
```
- Stage file(s) in your local repository to git.
```
git add .
```
- Commit the files that you've staged in your local repository.
```
git commit -m "initial commit"
```
- Create a new **private** github repo here: - [Github new repo](https://github.com/new) Don't initialize with a **README** or **.gitignore** files
- At the top of your repository on GitHub.com's Quick Setup page, copy the remote repository URL.
- On the same terminal/git bash, add the URL for the remote repository where your local repository will be pushed.
```
git remote add origin <REMOTE_URL>
git remote -v
git push -u origin master
```
- Run the above comands one at a time

- Show all branches by running the following command. As of now you should have only one branch: **master**
```
git branch -a
```
- Switch to master
```
git checkout master
```
- Create a local branch
```
git checkout -b new-BRANCH master # This command will create a new branch and switch to that branch
```
- Create a file in the new branch and name it as **practice.txt**
- Save the following in the file
```
Lorem ipsum dolor, sit amet consectetur adipisicing elit.
```
- See if there are any changes that need to be committed
```
git status
```
- Stage file(s) in your local repository to git.
```
git add .
```
- Commit the files that you've staged in your local repository.
```
git commit -m "practice example"
```
- Push new branch to remote server (**github**)
```
git push -u origin new-BRANCH
```
- Switch to master
```
git checkout master
```
- Switch to your branch
```
git checkout new-BRANCH
```
- **You might notice that the file practice.txt disappears when you switch to master then reappears when you switch back to your new branch. Thats how git works.**
- End of git branching practice
- Please invite me as a collaborator on github on the new private repo you created so that I can view your progress. My github **username** is **gdkimaiyo**
