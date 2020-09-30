
#####       Modelling       #####

inning1 = readRDS("C:/Users/Administrator/Desktop/inning1.rds")

View(inning1[which(inning1$X == 0),])
inning1 = inning1[-c(which(inning1$X == 0)),]
saveRDS(inning1,"C:/Users/Administrator/Desktop/inning1.rds")

I = max(inning1$batsman.id)
J = max(inning1$bowler.id)
N = dim(inning1)[1]
K = 6*9

inning1$L = numeric(dim(inning1)[1])
for (n in 1:N){
  inning1$L[n] = 1 + ifelse(inning1$wicket[n]>3,3,0) +  ifelse(inning1$wicket[n]>6,3,0) + ifelse(inning1$ball.no.[n]>36,1,0) + ifelse(inning1$ball.no.[n]>96,1,0)
}

data = list(I = I, J = J,X = inning1$X,N = N,L = inning1$L, batsmanid = inning1$batsman.id,bowlerid = inning1$bowler.id)
parameters = c("u1","u2","a","d")
inits = function(){
  list(u1 = rnorm(I,0,2),u2 = rnorm(J,0,2),z = rnorm(K,0,2),delta1 = runif(1,0,1),delta2 = runif(1,0,1),t = rgamma(1,1,2),s = rgamma(1,1,2))
}

library(R2WinBUGS)
inning1.model = bugs(data,inits,parameters,model.file = "C:/Program Files/WinBUGS14/inning1.odc",n.chains = 1, n.iter = 20000,n.burnin = 10000,n.thin = 10,codaPkg = T,bugs.directory = "C:/Program Files/WinBUGS14/")

codaobject = R2WinBUGS::read.bugs(inning1.model)
plot(codaobject)

View(codaobject[[1]])

save.image()
print(inning1.model)

#mean & sd of parameters
mean = lapply(codaobject,function(x){colMeans(x)})
mean = mean[[1]]
sd = lapply(codaobject,function(x){ apply(x,2,sd)})
sd = sd[[1]]

#extracting parameters from inning1.model
View(cbind(mean,1:length(mean)))

a = array(dim = c(9,8))
for (l in 1:9) {
  for(k in 1:8) {
    a[l,k] = mean[(l-1)*8 + k]    
  }
}

d = numeric()
d[1] = d[2] = d[3] = 0
for (l in 1:6) {
  d[l+3] = mean[l+72]
}

u1 = numeric()
for (i in 1:I) {
  u1[i] = mean[79+i]
}

u2 = numeric()
for (j in 1:J) {
  u2[j] = mean[594+j]
}


#calculating probabilities of each outcome in 1st inning using parameters 
p = array(dim = c(I,J,9,7))
for (i in 1:I) {
  for (j in 1:J) {
    for (l in 1:9) {
      for (k in 1:7){
        p[i,j,l,k] = 1/(1+exp(-(a[l,k+1] - u1[i] + u2[j] - d[l]))) - 1/(1+exp(-(a[l,k] - u1[i] + u2[j]-d[l])))
      }
    }
  }
}

#####    Wide and no balls    #####

wides = readRDS("C:/Users/Administrator/Desktop/wides.rds")
str(wides)
dataset = readRDS("C:/Users/Administrator/Desktop/dataset.rds")

t.wides_no = length(wides)        #total wide or no balls in all the matches considered
t.balls = dim(dataset)[1]         #total balls in all the matches considered
v = t.wides_no/t.balls            #probability of wide or no ball

#probability of outcome on a wide or no ball
pw = numeric(length = 7)          
for (i in 1:7) {
  n[i] = 0
  for (j in 1:t.wides_no) {
    n[i] = n[i] + ifelse(wides[j] == i,1,0)
  }
  pw[i] = n[i]/t.wides_no
}

dataset = readRDS("C:/Users/Administrator/Desktop/dataset.rds")
str(dataset)
id1 = data.frame(dataset$batsman,dataset$batsman.id,dataset$bowler,dataset$bowler.id)
library(dplyr)
id.set_batsman = data.frame(distinct(dataset,dataset$batsman),distinct(dataset,dataset$batsman.id))
id.set_bowler = data.frame(distinct(dataset,dataset$bowler),distinct(dataset,dataset$bowler.id))
View(id.set_batsman)
View(id.set_bowler)

save.image()

#####      SIMULATION - first inning     #####

wickets1 = 0
runs1 = 0
b0 = 1
batsman_order1 = numeric(11-wickets1)             # order if batting for remaining batsman
bowling_order1 = numeric(20- floor((b0-1)/6))     # order of bowling in remaining overs
X1 = numeric(120-b0+1)                            # vector containing output of remaining balls 
Y1 = numeric(120-b0+1)                            # vector containing output of remaining balls if wide/no ball
q1 = numeric(120-b0+1)

batsman_order1 = c(505,29,53,46,65,136,507,467,477,440,101)
bowling_order1 = c(16,385,16,24,385,16,71,28,71,28,71,24,28,71,28,24,385,16,385,24)

for(b in b0:120){
  if(wickets1 == 10){
    X1[b] = NA
  }else{
    check = 1
    while (check == 1) {
      u = runif(1)
      if(u < v){                     #wide or no ball
        
        runs1 = runs1+1
        
        random = runif(1)
        Y1[b] = 1 + ifelse(random > pw[1],1,0) + ifelse(random > (pw[1]+pw[2]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]+pw[5]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]+pw[5]+pw[6]),1,0)
        
        if(Y1[b] == 1){
          wickets1 = wickets1+1
        }
        if(Y1[b] == 3){
          runs1 = runs1+1
        }
        if(Y1[b] == 4){
          runs1 = runs1+2
        }
        if(Y1[b] == 5){
          runs1 = runs1+3
        }
        if(Y1[b] == 6){
          runs1 = runs1+4
        }
        if(Y1[b] == 7){
          runs1 = runs1+6
        }
        check = 1
        
      }else{
        
        #batsman to face the delivered ball
        
        if(b == b0){
          q1[b] = batsman_order1[1]
        }else{
          if(((b-1)/6 - as.integer((b-1)/6)) == 0){       #First ball of any over
            if(X1[b-1] == 1){                             #Wicket on last ball
              out_batsman = q1[which(X1[1:(b-1)] == 1)]
              batsman_batted = batsman_order1[1:(wickets1+2)]
              batsman_field = setdiff(batsman_batted,out_batsman)
              q1[b] = setdiff(batsman_field,q1[b-1])
            }else{
              if(X1[b-1] == 3 | X1[b-1] == 5){            #Batsmen rotated places while running btw wickets on last ball
                q1[b] = q1[b-1]
              }else{
                out_batsman = q1[which(X1[1:(b-1)] == 1)]
                batsman_batted = batsman_order1[1:(wickets1+2)]
                batsman_field = setdiff(batsman_batted,out_batsman)
                q1[b] = setdiff(batsman_field,q1[b-1])
              }
            }
          }else{
            if(X1[b-1] == 1){                             #Wicket on last ball
              q1[b] = batsman_order1[wickets1+2]
            }else{
              if(X1[b-1] == 3 | X1[b-1] == 5){            #Batsmen rotated places while running btw wickets on last ball
                out_batsman = q1[which(X1[1:(b-1)] == 1)]
                batsman_batted = batsman_order1[1:(wickets1+2)]
                batsman_field = setdiff(batsman_batted,out_batsman)
                q1[b] = setdiff(batsman_field,q1[b-1])
              }else{
                q1[b] = q1[b-1]
              }
            }
          }  
        }
        
        #bowler to ball 
        j = bowling_order1[ceiling((b-b0+1)/6)]
        
        l = 1 + ifelse(wickets1>3,3,0) +  ifelse(wickets1>6,3,0) + ifelse(b>36,1,0) + ifelse(b>96,1,0)
        
        random = runif(1)
        X1[b] = 1 + ifelse(random > p[q1[b],j,l,1],1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]+p[q1[b],j,l,5]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]+p[q1[b],j,l,5]+p[q1[b],j,l,6]),1,0)
        
        if(X1[b] == 3){
          runs1 = runs1+1
        }
        if(X1[b] == 4){
          runs1 = runs1+2
        }
        if(X1[b] == 5){
          runs1 = runs1+3
        }
        if(X1[b] == 6){
          runs1 = runs1+4
        }
        if(X1[b] == 7){
          runs1 = runs1+6
        }
        if(X1[b] == 1){
          wickets1 = wickets1+1
        }
        check = 0
        
      }
    }
  }
}

ball = rep(1:6, times = 20)
over = ceiling((1:120)/6)
inning1.sim = data.frame(X1,ball,over)
inning1.sim$is.wicket = numeric(length(X1))
inning1.sim$is.wicket[which(inning1.sim$X1 == 1)] = 1
inning1.sim$runs = numeric(length(X1))
inning1.sim$runs[which(inning1.sim$X1 == 3)] = 1
inning1.sim$runs[which(inning1.sim$X1 == 4)] = 2
inning1.sim$runs[which(inning1.sim$X1 == 5)] = 3
inning1.sim$runs[which(inning1.sim$X1 == 6)] = 4
inning1.sim$runs[which(inning1.sim$X1 == 7)] = 6
View(inning1.sim)
wides1.sim = data.frame(Y1,ball,over)
wides1.sim$is.wide.noball = numeric(length(Y1))
wides1.sim$is.wide.noball[which(wides1.sim$Y1 != 0)] = "Y"
wides1.sim$is.wide.noball[which(wides1.sim$Y1 == 0)] = "N"
wides1.sim$scored = numeric(length(Y1))
wides1.sim$scored[which(wides1.sim$is.wide.noball == "Y" & wides1.sim$Y1 == 3)] = 1
wides1.sim$scored[which(wides1.sim$is.wide.noball == "Y" & wides1.sim$Y1 == 4)] = 2
wides1.sim$scored[which(wides1.sim$is.wide.noball == "Y" & wides1.sim$Y1 == 5)] = 3
wides1.sim$scored[which(wides1.sim$is.wide.noball == "Y" & wides1.sim$Y1 == 6)] = 4
wides1.sim$scored[which(wides1.sim$is.wide.noball == "Y" & wides1.sim$Y1 == 7)] = 6
wides1.sim$extra = numeric(length(Y1))
wides1.sim$extra[which(wides1.sim$is.wide.noball == "Y")] = 1
wides1.sim$runs = wides1.sim$extra + wides1.sim$scored
wides1.sim$is.wicket = numeric(length(Y1))
wides1.sim$is.wicket[which(wides1.sim$Y1 == 1)] = 1
View(wides1.sim)

saveRDS(inning1.sim,"innings1.sim.rds")
saveRDS(wides1.sim,"wides1.sim.rds")

#####      SIMULATION - second inning     #####

#Duckworth-Lewis resources table (ball by ball)
library(haven)
resource.table = read.csv("C:/Users/Administrator/Downloads/twenty20.csv",header = T)
View(resource.table)
resource.table = resource.table[-1,]
resource.table = resource.table[,-c(1,2)]
resource.table = rbind(resource.table,numeric(10))
resource.table = cbind(resource.table,numeric(121))
rownames(resource.table) = 120:0
colnames(resource.table) = 10:0
r = resource.table/100
View(r)

#Resources lost due to wicket
y = array(dim = c(121,10))
for (i in 1:10){
  y[,i] = r[,i] - r[,(i+1)]
}
y = as.data.frame(y)
colnames(y) = 1:10
y = y[-121,]
View(y)

#Resources lost due to delivered ball
x = array(dim = c(120,11))
for (i in 1:120) {
  x[i,] = as.numeric(r[i,]) - as.numeric(r[(i+1),])
}
x = as.data.frame(x)
colnames(x) = 1:10
x = x[,-11]
View(x)

f = runs1
wickets2 = 0
runs2 = 0
b0 = 1
batsman_order2 = numeric(11-wickets2)             # order if battinng for remaining batsman
bowling_order2 = numeric(20- floor((b0-1)/6))     # order of bowling in remaining overs
X2 = numeric(120-b0+1)                            # vector containing output of remaining balls 
Y2 = numeric(120-b0+1)                            # vector containing output of remaining balls if wide/no ball
q2 = numeric(120-b0+1)

batsman_order2 = c(290,11,74,499,505,10,31,108,142,77,78)
bowling_order2 = c(78,366,78,366,337,385,103,337,108,337,108,337,78,385,108,366,78,385,366,385)

for(b in b0:120){
  if(wickets2 == 10){
    X2[b] = NA
  }else{
    check = 1
    while (check == 1) {
      u = runif(1)
      if(u < v){                      #wide or no ball
        
        runs2 = runs2+1
        
        random = runif(1)
        Y2[b] = 1 + ifelse(random > pw[1],1,0) + ifelse(random > (pw[1]+pw[2]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]+pw[5]),1,0) + ifelse(random > (pw[1]+pw[2]+pw[3]+pw[4]+pw[5]+pw[6]),1,0)
        
        if(Y2[b] == 1){
          wickets2 = wickets2+1
        }
        if(Y2[b] == 3){
          runs2 = runs2+1
        }
        if(Y2[b] == 4){
          runs2 = runs2+2
        }
        if(Y2[b] == 5){
          runs2 = runs2+3
        }
        if(Y2[b] == 6){
          runs2 = runs2+4
        }
        if(Y2[b] == 7){
          runs2 = runs2+6
        }
        check = 1
        
      }else{
        
        #batsman to face the delivered ball
        if(b == b0){
          q2[b] = batsman_order2[1]
        }else{
          if(((b-1)/6 - as.integer((b-1)/6)) == 0){                 #First ball of an over
            if(X2[b-1] == 1){                                       #wicket on last ball
              out_batsman = q2[which(X2[1:(b-1)] == 1)]
              batsman_batted = batsman_order2[1:(wickets2+2)]
              batsman_field = setdiff(batsman_batted,out_batsman)
              q2[b] = setdiff(batsman_field,q2[b-1])
            }else{
              if(X2[b-1] == 3 | X2[b-1] == 5){                      # Batsman rotated places while running between wickets
                q2[b] = q2[b-1]
              }else{
                out_batsman = q2[which(X2[1:(b-1)] == 1)]
                batsman_batted = batsman_order2[1:(wickets2+2)]
                batsman_field = setdiff(batsman_batted,out_batsman)
                q2[b] = setdiff(batsman_field,q2[b-1])
              }
            }
          }else{
            if(X2[b-1] == 1){                                       #wicket on last ball
              q2[b] = batsman_order2[wickets2+2]
            }else{
              if(X2[b-1] == 3 | X2[b-1] == 5){                      # Batsman rotated places while running between wickets
                out_batsman = q2[which(X2[1:(b-1)] == 1)]
                batsman_batted = batsman_order2[1:(wickets2+2)]
                batsman_field = setdiff(batsman_batted,out_batsman)
                q2[b] = setdiff(batsman_field,q2[b-1])
              }else{
                q2[b] = q2[b-1]
              }
            }
          }  
        }
        
        #Bowler to ball
        j = bowling_order2[ceiling((b-b0+1)/6)]
        
        #estimation of probability of outcome in 2nd inning
        
        p2 = numeric(7)
        l = 1 + ifelse(wickets2>3,3,0) +  ifelse(wickets2>6,3,0) + ifelse(b>36,1,0) + ifelse(b>96,1,0)
        E1 = p[q2[b],j,l,3] + 2*p[q2[b],j,l,4] + 3*p[q2[b],j,l,5] + 4*p[q2[b],j,l,6] + 6*p[q2[b],j,l,7]
        E2 = x[b,(wickets2+1)] + y[b,(wickets2+1)]*p[q2[b],j,l,1]
        delta = E2/(E2 + y[b,(wickets2+1)]*(1-p[q2[b],j,l,1]-p[q2[b],j,l,2]))
        
        c = as.numeric(r[b,(wickets2+1)]*E1/((f-runs2+1)*E2))
        p2[2] = c*p[q2[b],j,l,2]
        p2[1] = p[q2[b],j,l,1] + delta*p[q2[b],j,l,2]*(1-c)
        for (k in 3:7) {
          p2[k] = ((1-p[q2[b],j,l,1]-(c + delta*(1-c))*p[q2[b],j,l,2])/(1-p[q2[b],j,l,1]-p[q2[b],j,l,2]))*p[q2[b],j,l,k]
        }
        
        
        random = runif(1)
        X2[b] = 1 + ifelse(random > p2[1],1,0) + ifelse(random > (p2[1]+p2[2]),1,0) + ifelse(random > (p2[1]+p2[2]+p2[3]),1,0) + ifelse(random > (p2[1]+p2[2]+p2[3]+p2[4]),1,0) + ifelse(random > (p2[1]+p2[2]+p2[3]+p2[4]+p2[5]),1,0) + ifelse(random > (p2[1]+p2[2]+p2[3]+p2[4]+p2[5]+p2[6]),1,0)
        
        if(X2[b] == 3){
          runs2 = runs2+1
        }
        if(X2[b] == 4){
          runs2 = runs2+2
        }
        if(X2[b] == 5){
          runs2 = runs2+3
        }
        if(X2[b] == 6){
          runs2 = runs2+4
        }
        if(X2[b] == 7){
          runs2 = runs2+6
        }
        if(X2[b] == 1){
          wickets2 = wickets2+1
        }
        check = 0
        
      }
    }
  }
}

ball = rep(1:6, times = 20)
over = ceiling((1:120)/6)
inning2.sim = data.frame(X2,ball,over)
inning2.sim$is.wicket = numeric(length(X2))
inning2.sim$is.wicket[which(inning2.sim$X2 == 1)] = 1
inning2.sim$runs = numeric(length(X2))
inning2.sim$runs[which(inning2.sim$X2 == 3)] = 1
inning2.sim$runs[which(inning2.sim$X2 == 4)] = 2
inning2.sim$runs[which(inning2.sim$X2 == 5)] = 3
inning2.sim$runs[which(inning2.sim$X2 == 6)] = 4
inning2.sim$runs[which(inning2.sim$X2 == 7)] = 6
View(inning2.sim)
wides2.sim = data.frame(Y2,ball,over)
wides2.sim$is.wide.noball = numeric(length(Y2))
wides2.sim$is.wide.noball[which(wides2.sim$Y2 != 0)] = "Y"
wides2.sim$is.wide.noball[which(wides2.sim$Y2 == 0)] = "N"
wides2.sim$scored = numeric(length(Y2))
wides2.sim$scored[which(wides2.sim$is.wide.noball == "Y" & wides2.sim$Y2 == 3)] = 1
wides2.sim$scored[which(wides2.sim$is.wide.noball == "Y" & wides2.sim$Y2 == 4)] = 2
wides2.sim$scored[which(wides2.sim$is.wide.noball == "Y" & wides2.sim$Y2 == 5)] = 3
wides2.sim$scored[which(wides2.sim$is.wide.noball == "Y" & wides2.sim$Y2 == 6)] = 4
wides2.sim$scored[which(wides2.sim$is.wide.noball == "Y" & wides2.sim$Y2 == 7)] = 6
wides2.sim$extra = numeric(length(Y2))
wides2.sim$extra[which(wides2.sim$is.wide.noball == "Y")] = 1
wides2.sim$runs = wides2.sim$extra + wides2.sim$scored
wides2.sim$is.wicket = numeric(length(Y2))
wides2.sim$is.wicket[which(wides2.sim$Y2 == 1)] = 1
View(wides2.sim)

saveRDS(inning2.sim,"innings2.sim.rds")
saveRDS(wides2.sim,"wides2.sim.rds")








