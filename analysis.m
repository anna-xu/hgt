%% NOTES
% note: feedback is reversed to look at negative fb effects (i.e., FB = 1 means negative fb)

%% Confidence ratings were decoupled from accuracy

confAcc = fitlme(allSubData,...
    'Confidence ~ 1 + zAccuracy + (1 + zAccuracy|SubID)',...
    'FitMethod','REML') ;
[~,~,confAcc_results] = fixedEffects(confAcc, 'DFmethod', 'satterthwaite')

% confirming these results stay even if we use binned accuracy
confBinAcc = fitlme(allSubData,...
    'Confidence ~ 1 + binAccuracy + (1 + binAccuracy|SubID)',...
    'FitMethod','REML') ;
[~,~,confBinAcc_results] = fixedEffects(confBinAcc, 'DFmethod', 'satterthwaite')

%% Manipulation Check: Effort output (grip force) varies with both effort demands and time on task

effort_model = fitlme(allSubData,...
    'AbsEffort ~ 1 + zDifficulty * zTrialNum_Session + (1 + zDifficulty * zTrialNum_Session|SubID)',...
    'FitMethod','REML') ;
[~,~,effort_model_results] = fixedEffects(effort_model, 'DFmethod', 'satterthwaite')

undershoot_when_miss_bias = fitlme(allSubData,...
    'UndershootWhenMissPct ~ 1 + (1|SubID)',...
    'FitMethod','REML') ;
[~,~,undershoot_when_miss_bias_res] = fixedEffects(undershoot_when_miss_bias, 'DFmethod', 'satterthwaite')

% average accuracy and time in box
allSubData.derivedTimeInBox = allSubData.MeanTrialAcc * 2; % 2 is for 2 seconds (time after offset)

derivedTimeInBox_mdl = fitlme(allSubData,...
    'derivedTimeInBox ~ 1 + (1|SubID)',...
    'FitMethod','REML') ;
[~,~,derivedTimeInBox_mdl_res] = fixedEffects(derivedTimeInBox_mdl, 'DFmethod', 'satterthwaite')

average_acc = fitlme(allSubData,...
    'binAccuracy ~ 1 + (1|SubID)',...
    'FitMethod','REML') ;
[~,~,average_acc_res] = fixedEffects(average_acc, 'DFmethod', 'satterthwaite')

%% Measure validation: Fatigue and perceived exertion are tied to dissociable aspects of effort output

borg_time_effort = fitlme(allSubData,...
    'Borg ~ 1 + zTrialNum_Session * zAbsEffort + zConfidence + zTired + (1 + zTrialNum_Session * zAbsEffort + zConfidence + zTired|SubID)',...
    'FitMethod','REML') ;
[~,~,borg_time_effort_results] = fixedEffects(borg_time_effort, 'DFmethod', 'satterthwaite')

fatigue_time_effort = fitlme(allSubData,...
    'Tired ~ 1 + zTrialNum_Session * zAbsEffort + zConfidence + zBorg + (1 + zTrialNum_Session * zAbsEffort + zConfidence + zBorg|SubID)',...
    'FitMethod','REML') ;
[~,~,fatigue_time_effort_results] = fixedEffects(fatigue_time_effort, 'DFmethod', 'satterthwaite')

% crossover interaction

borgFatigueInteraction;

% Effort and performance over time

% Accuracy is continuous
accuracy_time = fitlme(allSubData,...,
    'zAccuracy ~ 1  + zTrialNum_Session + (1 + zTrialNum_Session|SubID)',...
    'FitMethod','REML') ;
[~,~,accuracy_time_results] = fixedEffects(accuracy_time, 'DFmethod', 'satterthwaite')

%% Fatigue increases more rapidly when receiving negative relative to positive feedback
fatigue_time_fb = fitlme(allSubData,...
    'Tired ~ 1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zBorg |SubID)',...
    'FitMethod','REML') ;
[~,~,fatigue_time_fb_results] = fixedEffects(fatigue_time_fb, 'DFmethod', 'satterthwaite')

borg_effort_fb = fitlme(allSubData,...
    'Borg ~ 1 + FeedbackCondition * zAbsEffort + zAbsEffort * zTrialNum_Session + zConfidence + zTired + (1 + FeedbackCondition * zAbsEffort + zAbsEffort * zTrialNum_Session + zConfidence + zTired |SubID)',...
    'FitMethod','REML') ;
[~,~,borg_effort_fb_results] = fixedEffects(borg_effort_fb, 'DFmethod', 'satterthwaite')

% supplementary table 2

borg_time_fb = fitlme(allSubData,...
    'Borg ~ 1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zTired + (1 + FeedbackCondition * zTrialNum_Session + zAbsEffort * zTrialNum_Session + zConfidence + zTired |SubID)',...
    'FitMethod','REML') ;
[~,~,borg_time_fb_results] = fixedEffects(borg_time_fb, 'DFmethod', 'satterthwaite')

fatigue_effort_fb = fitlme(allSubData,...
    'Tired ~ 1 + FeedbackCondition * zAbsEffort + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zAbsEffort + zAbsEffort * zTrialNum_Session + zConfidence + zBorg |SubID)',...
    'FitMethod','REML') ;
[~,~,fatigue_effort_fb_results] = fixedEffects(fatigue_effort_fb, 'DFmethod', 'satterthwaite')

% supplementary table 3

effort_feedback_model = fitlme(allSubData,...,
    'AbsEffort ~ 1 + FeedbackCondition * zDifficulty * zTrialNum_Session + (1 + FeedbackCondition * zDifficulty * zTrialNum_Session|SubID)',...
    'FitMethod','REML') ;
[~,~,effort_feedback_model_results] = fixedEffects(effort_feedback_model, 'DFmethod', 'satterthwaite')

% supplementary table 4

fatigue_time_conditionOrder = fitlme(allSubData,...
    'Tired ~ 1 + FeedbackCondition * zTrialNum_Session + FeedbackCondition * ConditionOrder + zAbsEffort * zTrialNum_Session + zConfidence + zBorg + (1 + FeedbackCondition * zTrialNum_Session + FeedbackCondition * ConditionOrder + zAbsEffort * zTrialNum_Session + zConfidence + zBorg |SubID)',...
    'FitMethod','REML') ;
[~,~,fatigue_time_conditionOrder_results] = fixedEffects(fatigue_time_conditionOrder, 'DFmethod', 'satterthwaite')

% supplementary table 5

effort_fb_conditionOrder = fitlme(allSubData,...,
    'AbsEffort ~ 1 + FeedbackCondition * zDifficulty*zTrialNum_Session + FeedbackCondition*ConditionOrder + (1 +  FeedbackCondition * zDifficulty*zTrialNum_Session + FeedbackCondition*ConditionOrder|SubID)',...
    'FitMethod','REML') ;
[~,~,effort_fb_conditionOrder_results] = fixedEffects(effort_fb_conditionOrder, 'DFmethod', 'satterthwaite')

% figures

% predicted data
feedback_figure('Fatigue', fatigue_time_fb, allSubData, 'Time')
feedback_figure('Perceived Exertion', borg_effort_fb, allSubData, 'Grip Force')

% figures

feedback_figure('Grip Force', effort_feedback_model, allSubData, 'Effort Demands');
feedback_figure('Grip Force', effort_feedback_model, allSubData, 'Time');

%% Confidence reflects cumulative feedback, despite both being decoupled from performance

confidence_fb = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback * FeedbackCondition + (1 + CumulativeFeedback * FeedbackCondition|SubID)',...
    'FitMethod','REML') ;
[~,~,results] = fixedEffects(confidence_fb, 'DFmethod', 'satterthwaite')

% figures

feedback_figure('Confidence', confidence_fb, allSubData, 'Cumulative Feedback');

% Break confidence into positive and negative feedback blocks
conf_posFB = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback + (1 + CumulativeFeedback|SubID)',...
    'FitMethod','REML', 'Exclude', allSubData.FeedbackCondition==categorical(1)) ;
[~,~,confidence_posFB_results] = fixedEffects(conf_posFB, 'DFmethod', 'satterthwaite')

conf_negFB = fitlme(allSubData,...
    'Confidence ~ 1 + CumulativeFeedback + (1 + CumulativeFeedback|SubID)',...
    'FitMethod','REML', 'Exclude', allSubData.FeedbackCondition==categorical(0)) ;
[~,~, conf_negFB_results] = fixedEffects(conf_negFB, 'DFmethod', 'satterthwaite')

% difference between magnitude of slopes (from conf_posFB and conf_negFB)
beta1 = 0.0055721;
beta2 = -0.0093351; 
seBeta1 = 0.0013426; 
seBeta2 = 0.0017558;

beta_coeff_test(beta1, abs(beta2), seBeta1, seBeta2)
z=-1.7025; % from above
two_tailed_p = normcdf(z) * 2;

%% supplementary
% supplementary table

% Figure S1
rawPlotAcc(allSubData, 'Confidence', 'Confidence')

conf_effort = fitlme(allSubData,...
    'Confidence ~ 1 + zAbsEffort + (1 + zAbsEffort|SubID)',...
    'FitMethod','REML') ;
[~,~,conf_effort_results] = fixedEffects(conf_effort, 'DFmethod', 'satterthwaite')

conf_fatigue = fitlme(allSubData,...
    'Confidence ~ 1 + zTired + (1 + zTired|SubID)',...
    'FitMethod','REML') ;
[~,~,conf_fatigue_results] = fixedEffects(conf_fatigue, 'DFmethod', 'satterthwaite')

conf_perceived_exertion = fitlme(allSubData,...
    'Confidence ~ 1 + zBorg + (1 + zBorg|SubID)',...
    'FitMethod','REML') ;
[~,~,conf_perceived_exertion_results] = fixedEffects(conf_perceived_exertion, 'DFmethod', 'satterthwaite')

% See above section for confidence and accuracy model

%% Methods -- Feedback Eligibility

% Rate of eligible feedback trials
% Feedback eligibility criteria uses a heuristic that roughly centers
% around the center of the target difficulty by using half of the
% y-coordinate at the bottom of the band as a new reference value
allSubData.precision_from_bottom_half = abs(allSubData.MercLvl - (allSubData.bottomHeight/2));
allSubData.actual_eligibility = (allSubData.precision_from_bottom_half >= 50) & ...
                                  (allSubData.precision_from_bottom_half <= 300);
subject_actual_eligibility = groupsummary(allSubData, 'SubID', 'mean', 'actual_eligibility');
overallMeanCount = mean(subject_actual_eligibility.mean_actual_eligibility)

% see above section for average accuracy

%% Additional Tests -- Feedback Count
% see script check_fb_exclusions.m for models testing stability of results
% when excluding extreme cases of missingness

fb_count_table = grpstats(allSubData, {'SubID', 'FeedbackCondition'}, 'sum', 'DataVars', {'IsFeedback'});

% avg percent missing per feedback condition blocks
fb_count_table.diffFromExpected = 33 - fb_count_table.sum_IsFeedback;
fb_count_table.percMissing = fb_count_table.diffFromExpected / 33;
mean(fb_count_table.percMissing)

% paired t-test

wideTbl = unstack(fb_count_table, 'sum_IsFeedback', 'FeedbackCondition');
wideTbl.NegativeFBCount = wideTbl.x1;
wideTbl.PositiveFBCount = wideTbl.x0;

% 2. Extract the paired columns into vectors
X = wideTbl.NegativeFBCount;
Y = wideTbl.PositiveFBCount;

% 3. Run the two-tailed paired t-test
[h, p, ci, stats] = ttest(X, Y);

% 4. Display the results
disp(stats);
fprintf('p-value = %.4f\n', p);

%% Figure for feedback count table
% Note: you have to zoom in on the matlab fig so that the boxes are more
% square

figure();
h=histogram2(X,Y);
view(2);
h.DisplayStyle='tile';
clim([0,5]);
xlim([0,33]);
xlabel('Negative Feedback Count');
ylabel('Positive Feedback Count');
ylim([0,33]);
colormap(jet);
c = colorbar;
c.Label.String = 'Number of Participants';
c.Label.Rotation = 270; 
c.Ticks = [0, 1, 2, 5];
c.TickLabels = {'0', '1', '2', '49'};
c.FontSize = 14;
set(gca,'FontSize', 14);
grid off;
box off;