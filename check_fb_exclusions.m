%% Fatigue 

fatigue_time_fb1 = fitlme(allSubData,...
     'Tired ~ 1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg|SubID)',...
     'FitMethod','REML', 'Exclude', (allSubData.NegMinusPosFBCount > 6));
[~,~,fatigue_time_fb1_results] = fixedEffects(fatigue_time_fb1, 'DFmethod', 'satterthwaite')

fatigue_time_fb2 = fitlme(allSubData,...
     'Tired ~ 1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg|SubID)',...
     'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8));
[~,~,fatigue_time_fb2_results] = fixedEffects(fatigue_time_fb2, 'DFmethod', 'satterthwaite')

fatigue_time_fb3 = fitlme(allSubData,...
     'Tired ~ 1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg|SubID)',...
     'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8 | allSubData.NegMinusPosFBCount > 6));
[~,~,fatigue_time_fb_results3] = fixedEffects(fatigue_time_fb3, 'DFmethod', 'satterthwaite')

%% Grip Force

effort_feedback_model1 = fitlme(allSubData,...,
    'AbsEffort ~ 1 + FeedbackCondition * zDifficulty * zTrialNum_Session + (1 + FeedbackCondition * zDifficulty * zTrialNum_Session|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.NegMinusPosFBCount > 6)) ;
[~,~,effort_feedback_model_results1] = fixedEffects(effort_feedback_model1, 'DFmethod', 'satterthwaite')

effort_feedback_model2 = fitlme(allSubData,...,
    'AbsEffort ~ 1 + FeedbackCondition * zDifficulty * zTrialNum_Session + (1 + FeedbackCondition * zDifficulty * zTrialNum_Session|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8)) ;
[~,~,effort_feedback_model_results2] = fixedEffects(effort_feedback_model2, 'DFmethod', 'satterthwaite')

effort_feedback_model3 = fitlme(allSubData,...,
    'AbsEffort ~ 1 + FeedbackCondition * zDifficulty * zTrialNum_Session + (1 + FeedbackCondition * zDifficulty * zTrialNum_Session|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8 | allSubData.NegMinusPosFBCount > 6)) ;
[~,~,effort_feedback_model_results3] = fixedEffects(effort_feedback_model3, 'DFmethod', 'satterthwaite')

%% Confidence

confidence_fb1 = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback * FeedbackCondition + (1 + CumulativeFeedback * FeedbackCondition|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.NegMinusPosFBCount > 6)) ;
[~,~,results1] = fixedEffects(confidence_fb1, 'DFmethod', 'satterthwaite')

confidence_fb2 = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback * FeedbackCondition + (1 + CumulativeFeedback * FeedbackCondition|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8)) ;
[~,~,results2] = fixedEffects(confidence_fb2, 'DFmethod', 'satterthwaite')

confidence_fb3 = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback * FeedbackCondition + (1 + CumulativeFeedback * FeedbackCondition|SubID)',...
    'FitMethod','REML', 'Exclude', (allSubData.FBCoverageProp < 0.8 | allSubData.NegMinusPosFBCount > 6)) ;
[~,~,results] = fixedEffects(confidence_fb3, 'DFmethod', 'satterthwaite')