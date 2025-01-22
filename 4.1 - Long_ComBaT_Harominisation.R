# Make sure to install R because this is an R script
# Author: Diana Giraldo UAntwerpen, Hamza Khan, UHasselt/UMaastricht
# This script Harmonises the data using LongComBat. The inspiration is taken from https://github.com/jcbeer/longCombat
# Load packages
library(dplyr)
library(lubridate)
library(longCombat)
library(invgamma)
library(lme4)
library(ggplot2)

# Read list of features to harmonize
feats_for_combat <- readLines("....only_feats.txt")

#################################
# Read training data
#################################
train_df <- read.csv(".../NonHarmonised_Dataset.csv") %>%
  mutate(PATID = as.factor(PATID),
         Gender = as.factor(Gender),
         Age = Age_in_months/12,
         MRIpipeline = factor(MRIpipeline,
                              levels = c("CsT1", "C", "B", "A")))
# First level of MRIpipeline factor should be the best one

# # Check and handle NaN values in the specified features
# nan_columns <- which(colSums(is.na(train_df[, feats_for_combat])) > 0)
# # ! Nothing

# Convert MRI date and calculate months and days since first visit
train_df <- train_df %>%
  mutate(MRIdate = as.Date(MRIdate)) %>%
  group_by(PATID) %>%
  mutate(t0 = min(MRIdate),
         Month = round(time_length(interval(t0, MRIdate), "month")),
         days_since_first = round(time_length(interval(t0, MRIdate), "day")))%>%
  as.data.frame(.)

#################################
# Specs for modeling / long combat
#################################
idvar <- 'PATID'

#timevar <- "days_since_first"
timevar <- 'Month'  

batchvar <- 'MRIpipeline'

featurenames <- feats_for_combat
n_feat <- length(feats_for_combat)

random_effects <- '(1|PATID)' # Random intercept for each patient. Adjust if you want to include random slopes.

formula <- 'Age + Gender + Month' # known explanatory variables, including the time variable and possible interactions
#fixed_effects = 'Month'

#################################
# longcombat step-by-step
#################################

# number of batches
n_batch = nlevels(train_df[,batchvar])
# number of observations per batch
n_i = table(train_df[,batchvar])
batches = lapply(levels(train_df[,batchvar]), function(x) which(train_df[,batchvar]==x))

# 1. Fit linear model per feature, estimate sigmas and batch effects
method = 'REML'
all_fit <- list()
sigma_estimates <- rep(NA, n_feat)
batch_effects <- matrix(nrow=(n_batch-1), ncol = n_feat)

for (v in 1:n_feat){
  # Model fit
  lme_formula <- as.formula(paste0(featurenames[v], '~', formula, '+' , batchvar, '+', random_effects))
  lme_fit <- lme4::lmer(lme_formula, data=train_df, REML=TRUE, control=lme4::lmerControl(optimizer='bobyqa'))
  all_fit <- append(all_fit, lme_fit)
  # Estimate sigmas
  if (method == 'REML'){
    corr_estimates <- as.data.frame(lme4::VarCorr(lme_fit))
    sigma_estimates[v] <- corr_estimates[corr_estimates$grp=='Residual','sdcor']
  } else if (method == 'MSR'){
    resid <- residuals(lme_fit)
    sigma_estimates[v] <- sqrt((sum((resid-mean(resid))^2)/length(resid)))
  }
  # Estimate batch effects hat{gamma}_iv
  batch_effects[,v] <- lme4::fixef(lme_fit)[grep(batchvar, names(lme4::fixef(lme_fit)))]
  # Remove
  rm(corr_estimates, lme_fit, lme_formula)
}
rm(v)

# Adjust batch effects such that sum_i n_i * hat{gamma}_iv = 0
# weighted sum of estimated batch effects per feature
gamma1hat <- -(n_i[2:n_batch] %*% batch_effects)/sum(n_i)
batch_effects_adjusted <- sweep(batch_effects, 2, gamma1hat, FUN='+')
# add gamma1hat as the first batch effect
batch_effects_adjusted <- rbind(gamma1hat, batch_effects_adjusted)
rownames(batch_effects_adjusted) <- levels(train_df[,batchvar])
# # Check constraint
# tmp = n_i %*% batch_effects_adjusted
rm(batch_effects, gamma1hat)

# 2. Standardize features
feat_std <- matrix(nrow=nrow(train_df), ncol=n_feat)
colnames(feat_std) <- featurenames
for (v in 1:n_feat){
  f_predicted <- predict(all_fit[[v]])
  f_batch_effect <- batch_effects_adjusted[train_df[,batchvar],v] #batch_effects_expanded
  feat_std[,v] <- (train_df[,featurenames[v]] - f_predicted + f_batch_effect) / sigma_estimates[v]
  rm(f_predicted, f_batch_effect)
}
rm(v)

# 3. Estimation of hyperparameters using the method of moments
# Combat assumes that , for a given batch, the additive batch parameters across features (gammas) come from a common distribution, and similarly for batch scaling factors (deltas2)
# In this step the four parameters of those two distributions are estimated:
# additive batch effects ~ Normal(gamma_i, tau^2_i)
# multiplicative batch effects ~ InverseGamma(lambda_i, theta_i)

# Calculate sample mean and variance per batch/feature  
tmp_feat_std <- as.data.frame(feat_std) %>%
  bind_cols(select(train_df, all_of(batchvar))) %>%
  group_by_at(batchvar) 

# Hyper-params additive batch effects distribution
gammahat <- summarise(tmp_feat_std, across(everything(), mean))
gammabar <- rowMeans(gammahat[,featurenames])
tau2bar <- apply(gammahat[,featurenames], 1, var)

# Hyper-params multiplicative batch effects distribution
delta2hat <- summarise(tmp_feat_std, across(everything(), var))
Dbar <- rowMeans(delta2hat[,featurenames])
S2bar <- apply(delta2hat[,featurenames], 1, var)
lambdabar <- (Dbar^2 + 2*S2bar) / S2bar
thetabar <- (Dbar^3 + Dbar*S2bar) / S2bar

rm(tmp_feat_std, Dbar, S2bar)

# 4. empirical Bayes to estimate batch effects: gammastarhat, delta2starhat
n_iter=30
gammastarhat <- array(dim=c(n_batch, ncol=n_feat, (n_iter+1)))
delta2starhat <- array(dim=c(n_batch, ncol=n_feat, (n_iter+1)))

# Initial estimates
for (v in 1:n_feat){
  gammastarhat[,v,1] <- (((n_i * tau2bar * gammahat[,featurenames[v]]) + (delta2hat[,featurenames[v]] * gammabar))/((n_i * tau2bar) + delta2hat[,featurenames[v]]))[[1]]
  for(i in 1:n_batch){
    zminusgammastarhat2 <- sum((feat_std[batches[[i]],featurenames[v]] - gammastarhat[i,v,1])^2)
    delta2starhat[i,v,1] <- (thetabar[i] + 0.5*zminusgammastarhat2) / (n_i[i]/2 + lambdabar[i] - 1)
    rm(zminusgammastarhat2)
  }
}

for (ite in 1:n_iter){
  for (v in 1:n_feat){
    gammastarhat[,v,(ite+1)] <- (((n_i * tau2bar * gammahat[,featurenames[v]]) + (delta2starhat[,v,ite] * gammabar))/((n_i * tau2bar) + delta2starhat[,v,ite]))[[1]]
    for(i in 1:n_batch){
      zminusgammastarhat2 <- sum((feat_std[batches[[i]],featurenames[v]] - gammastarhat[i,v,ite])^2)
      delta2starhat[i,v,(ite+1)] <- (thetabar[i] + 0.5*zminusgammastarhat2) / (n_i[i]/2 + lambdabar[i] - 1)
    }
  }
}
gammastarhat_final <- gammastarhat[,,n_iter+1]
rownames(gammastarhat_final) <- levels(train_df[,batchvar])
delta2starhat_final <- delta2starhat[,,n_iter+1]
rownames(delta2starhat_final) <- levels(train_df[,batchvar])

rm(gammabar, tau2bar, lambdabar, thetabar, gammahat, delta2hat, gammastarhat, delta2starhat, batches)

# 5. Do Combat
feat_combat <- matrix(nrow=nrow(train_df), ncol=n_feat)
colnames(feat_combat) <- featurenames
for (v in 1:n_feat){
  f_gammastarhat <- gammastarhat_final[train_df[,batchvar],v]
  f_delta2starhat <- delta2starhat_final[train_df[,batchvar],v]
  f_predicted <- predict(all_fit[[v]])
  f_batch_effect <- batch_effects_adjusted[train_df[,batchvar],v] #batch_effects_expanded
  feat_combat[,v] <- (sigma_estimates[v]/sqrt(f_delta2starhat))*(feat_std[,v] - f_gammastarhat) + f_predicted - f_batch_effect
  rm(f_gammastarhat, f_delta2starhat, f_predicted, f_batch_effect)
}
train_combat <- cbind(train_df[,c(idvar, timevar, batchvar)], feat_combat)
colnames(train_combat) <- c(idvar, timevar, batchvar, featurenames) 
#colnames(train_combat) <- c(idvar, timevar, batchvar, paste0(featurenames, '.combat')) 

#################################
# Check harmonization training data
#################################
pca_pre <- prcomp(train_df[, featurenames], center = TRUE, scale. = TRUE)
pca_post <- prcomp(train_combat[,featurenames], center = TRUE, scale. = TRUE)

plot_df <- bind_rows(
  data.frame(PC1 = pca_pre$x[,1], PC2 = pca_pre$x[,2], Batch = train_df$MRIpipeline, Harm = "pre"),
  data.frame(PC1 = pca_post$x[,1], PC2 = pca_pre$x[,2], Batch = train_df$MRIpipeline, Harm = "post")
) %>%
  mutate(Harm = factor(Harm, levels = c("pre", "post"), labels = c("Pre harmonization", "Post harmonization")))

ggplot(plot_df, aes(x = PC1, y = PC2, color = Batch)) + 
  facet_grid(. ~ Harm) +
  geom_point() + 
  theme_minimal()  

#################################
# Read testing data
#################################
test_df <- read.csv("NonHarmonised_Test_df.csv") %>%
  mutate(PATID = as.factor(PATID),
         Gender = as.factor(Gender),
         Age = Age_in_months/12,
         MRIpipeline = factor(MRIpipeline,
                              levels = c("CsT1", "C", "A", "D")), # LEVELS SHOULD BE THE SAME AS IN TRAINING
         MRIdate = as.Date(MRIdate)) %>%
  group_by(PATID) %>%
  mutate(t0 = min(MRIdate),
         Month = round(time_length(interval(t0, MRIdate), "month")),
         days_since_first = round(time_length(interval(t0, MRIdate), "day")))%>%
  as.data.frame(.)

# Update models
update_all_fit <- list()
for (v in 1:n_feat){
  update_fit <- update(all_fit[[v]], data = bind_rows(train_df, test_df))
  update_all_fit <- append(update_all_fit, update_fit)
  rm(update_fit)
}

# Standardize new data
new_std <- matrix(nrow=nrow(test_df), ncol=n_feat)
colnames(new_std) <- featurenames
for (v in 1:n_feat){
  f_predicted <- predict(update_all_fit[[v]], newdata = test_df)
  f_batch_effect <- batch_effects_adjusted[test_df[,batchvar],v] #batch_effects_expanded
  new_std[,v] <- (test_df[,featurenames[v]] - f_predicted + f_batch_effect) / sigma_estimates[v]
  rm(f_predicted, f_batch_effect)
}

# Apply combat to new data
new_combat <- matrix(nrow=nrow(test_df), ncol=n_feat)
colnames(new_combat) <- featurenames
for (v in 1:n_feat){
  f_gammastarhat <- gammastarhat_final[test_df[,batchvar],v]
  f_delta2starhat <- delta2starhat_final[test_df[,batchvar],v]
  f_predicted <- predict(update_all_fit[[v]], newdata = test_df)
  f_batch_effect <- batch_effects_adjusted[test_df[,batchvar],v] #batch_effects_expanded
  new_combat[,v] <- (sigma_estimates[v]/sqrt(f_delta2starhat))*(new_std[,v] - f_gammastarhat) + f_predicted - f_batch_effect
  rm(f_gammastarhat, f_delta2starhat, f_predicted, f_batch_effect)
}
test_combat <- cbind(test_df[,c(idvar, timevar, batchvar)], new_combat)
colnames(test_combat) <- c(idvar, timevar, batchvar, featurenames) 
#colnames(test_combat) <- c(idvar, timevar, batchvar, paste0(featurenames, '.combat')) 

#################################
# Check harmonization training data
#################################
pca_pre <- prcomp(test_df[, featurenames], center = TRUE, scale. = TRUE)
pca_post <- prcomp(test_combat[,featurenames], center = TRUE, scale. = TRUE)

plot_df <- bind_rows(
  data.frame(PC1 = pca_pre$x[,1], PC2 = pca_pre$x[,2], Batch = test_df$MRIpipeline, Harm = "pre"),
  data.frame(PC1 = pca_post$x[,1], PC2 = pca_pre$x[,2], Batch = test_df$MRIpipeline, Harm = "post")
) %>%
  mutate(Harm = factor(Harm, levels = c("pre", "post"), labels = c("Pre harmonization", "Post harmonization")))

ggplot(plot_df, aes(x = PC1, y = PC2, color = Batch)) + 
  facet_grid(. ~ Harm) +
  geom_point() + 
  theme_minimal()  

#################################
# Check harmonization across training testing data
#################################
pca_pre <- prcomp(rbind(train_df[, featurenames], test_df[, featurenames]), center = TRUE, scale. = TRUE)
pca_post <- prcomp(rbind(train_combat[,featurenames], test_combat[,featurenames]), center = TRUE, scale. = TRUE)

plot_df <- bind_rows(
  data.frame(PC1 = pca_pre$x[,1], PC2 = pca_pre$x[,2], 
             Batch = c(train_df$MRIpipeline, test_df$MRIpipeline), 
             Set = c(rep("train", nrow(train_df)), rep("test", nrow(test_df))),
             Harm = "pre"),
  data.frame(PC1 = pca_post$x[,1], PC2 = pca_pre$x[,2], 
             Batch = c(train_df$MRIpipeline, test_df$MRIpipeline), 
             Set = c(rep("train", nrow(train_df)), rep("test", nrow(test_df))),
             Harm = "post")
) %>%
  mutate(Harm = factor(Harm, levels = c("pre", "post"), labels = c("Pre harmonization", "Post harmonization")),
         Set = factor(Set, levels = c("train", "test")))

ggplot(plot_df, aes(x = PC1, y = PC2, color = Set:Batch)) + 
  facet_grid(. ~ Harm) +
  geom_point() + 
  theme_minimal()  

# Combine harmonized features with specified non-harmonized columns for training data (PELT)
train_combat_combined <- cbind(train_df[, c('disability_progression',
                                            'Index_NH',
                                            'MRIdate',
                                            'Session')], 
                               train_combat[, featurenames])

# Save the combined training data
write.csv(train_combat_combined, "Path_to_save_Harmonised.csv", row.names = FALSE)

# Combine harmonized features with specified non-harmonized columns for testing data (ZMC)
test_combat_combined <- cbind(test_df[, c('disability_progression',
                                          't0',
                                          'Age_in_months',
                                          'EDSS_T0',
                                          'Gender',
                                          'Gender_F',
                                          'Gender_M',
                                          'Index_NH',
                                          'MRIdate',
                                          'Session')], 
                              test_combat[, featurenames])

# Save the combined testing data
write.csv(test_combat_combined, "Path_to_save.csv", row.names = FALSE)

###############################################################################

# PREV TEST.........................

###############################################################################


#################################
# longCombat() -- apply longitudinal ComBat
#################################
DF_combat <- longCombat(idvar='ID', 
                        timevar='Month',
                        batchvar=batch_variable,  
                        features=vol_feat, 
                        formula=fixed_effects,
                        ranef=random_effects,
                        data=DF)

DF_harmon <- DF_combat$data_combat
vol_feat_combat <- paste(vol_feat, "combat", sep = ".")
DF <- merge(DF, DF_harmon)


###############################################################################
# plot distributions of features
library("reshape2")
library("ggplot2")
X <- DF %>%
  melt(measure.vars = vol_feat)
pl = ggplot(X, aes(x = value, colour = MRIpipeline, fill = MRIpipeline)) +
  #geom_histogram(aes(y=..density..)) +
  geom_density(alpha=.2) +
  facet_wrap(vars(variable), nrow =6, scales = "free") +
  theme_bw()
pl


# plot distributions
X <- DF %>%
  melt(measure.vars = vol_feat_combat)
pl = ggplot(X, aes(x = value, colour = MRIpipeline, fill = MRIpipeline)) +
  geom_density(alpha=.2) +
  facet_wrap(vars(variable), nrow =6, scales = "free") +
  theme_bw()
pl
rm(X)


#################################
# plot trajectories before and after combat
#################################
sel_feat = vol_feat[39]
trajPlot(idvar='ID', 
         timevar='Month',
         feature=sel_feat, 
         batchvar='MRIpipeline',  
         data=DF,
         point.col=DF$MRIpipeline,
         title=paste(sel_feat, " before combat"))

trajPlot(idvar='ID', 
         timevar='Month',
         feature=paste(sel_feat, "combat", sep = "."), 
         batchvar='MRIpipeline',  
         data=DF,
         point.col=DF$MRIpipeline,
         title=paste(sel_feat, " after combat"))
