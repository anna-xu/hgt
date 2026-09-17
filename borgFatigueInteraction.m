%% Crossover interaction between Perceived Exertion & Fatigue

outcome = 'subjectExperienceRating';

allSubData.Difficulty = allSubData.zDifficulty;

% create a table for analyses
borgFatigueVals = [allSubData.Borg; allSubData.Tired];
borgFatigueDummy = [zeros(26928/2, 1); ones(26928/2,1)]; % 0 = Borg
difficulty = [allSubData.Difficulty; allSubData.Difficulty];
confidence = [zscoreNan(allSubData.Confidence); zscoreNan(allSubData.Confidence)];
effort = [zscore(allSubData.AbsEffort); zscore(allSubData.AbsEffort)];
time = [allSubData.zTrialNum_Session; allSubData.zTrialNum_Session];
subID = [allSubData.SubID; allSubData.SubID];

dummyTable = [borgFatigueVals'; borgFatigueDummy'; difficulty'; confidence'; effort'; time']';
borgFatigueHdr = {'BorgFatigueVal', 'isBorgFatigue', 'Difficulty', 'Confidence', 'Effort', 'Time'};
borgFatigueTbl = array2table(dummyTable);
borgFatigueTbl.Properties.VariableNames = borgFatigueHdr; 
borgFatigueTbl.SubID = subID;
borgFatigueTbl.cIsBorgFatigue = categorical(borgFatigueTbl.isBorgFatigue);

% crossover interaction model
borg_fatigue_model = fitlme(borgFatigueTbl,...
    'BorgFatigueVal ~ 1 + cIsBorgFatigue*Effort + cIsBorgFatigue*Time + Confidence + (1 + cIsBorgFatigue*Effort + cIsBorgFatigue*Time + Confidence|SubID)',...
    'FitMethod','REML') ;
[~,~,borg_fatigue_model_res] = fixedEffects(borg_fatigue_model, 'DFmethod', 'satterthwaite')

%% Plot raw values for each, discretized, for time
num_bins = 5;
% subset borg and fatigue data
fatigueData = borgFatigueTbl((borgFatigueTbl.isBorgFatigue==1),1:width(borgFatigueTbl));
borgData = borgFatigueTbl((borgFatigueTbl.isBorgFatigue==0),1:width(borgFatigueTbl));
% time plots
figure()
fatigueData_first = fatigueData((fatigueData.Time < 0), 1:width(fatigueData));
borgData_first = borgData((borgData.Time < 0), 1:width(borgData));

fatigueData_last = fatigueData((fatigueData.Time >= 0), 1:width(fatigueData));
borgData_last = borgData((borgData.Time >= 0), 1:width(borgData));

first_binned_var = fatigueData_first.Time;
[first_bins, first_binValues] = discretize(first_binned_var, num_bins);

last_binned_var = fatigueData_last.Time;
[last_bins, last_binValues] = discretize(last_binned_var, num_bins);

fatigueData_first.binnedTargets = first_bins;
borgData_first.binnedTargets = first_bins;

fatigueData_last.binnedTargets = last_bins;
borgData_last.binnedTargets = last_bins;

% group stats -- first half 

figure(1);
% plot fatigue
fatigueData_first_binnedTargets_sub = grpstats(fatigueData_first,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
fatigueData_first_binnedTargets_all = grpstats(fatigueData_first_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level
% actual plot
fatiguePlotFirst = plot(fatigueData_first_binnedTargets_all.binnedTargets, fatigueData_first_binnedTargets_all{:,3}, 'm', 'Marker', 'o');
fatiguePlotFirst.LineWidth=2;
hold on
eG=errorbar(fatigueData_first_binnedTargets_all{:,3}, fatigueData_first_binnedTargets_all{:,4}, '.');
eG.Color='m';
eG.LineWidth=2;
hold on

% plot borg

% group stats
borgData_first_binnedTargets_sub = grpstats(borgData_first,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
borgData_first_binnedTargets_all = grpstats(borgData_first_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level

% actual plot
borgPlotFirst = plot(borgData_first_binnedTargets_all.binnedTargets, borgData_first_binnedTargets_all{:,3}, 'b', 'Marker', 'o');
borgPlotFirst.LineWidth=2;
hold on
eG=errorbar(borgData_first_binnedTargets_all{:,3}, borgData_first_binnedTargets_all{:,4}, '.');
eG.Color='b';
eG.LineWidth=2;
hold on

% Make the graphs pretty
set(gca, 'FontSize', 12)
hold on
ylabel('Subjective Experience Rating', 'FontSize', 14)
hold on
xlabel('Trial (z-scored)', 'FontSize', 14)
hold on
set(gca, 'xtick', 1:num_bins, 'xticklabel', first_binValues(1:num_bins));
set(gca, 'ylim', [.25, .6])
set(gca,'TickDir','out');

box off;

saveas(gcf, 'raw_ratings_from_isBorgFatigue_split_first.png')

% group stats -- last half 

figure(2);
% plot fatigue
fatigueData_last_binnedTargets_sub = grpstats(fatigueData_last,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
fatigueData_last_binnedTargets_all = grpstats(fatigueData_last_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level
% actual plot
fatiguePlotFirst = plot(fatigueData_last_binnedTargets_all.binnedTargets, fatigueData_last_binnedTargets_all{:,3}, 'm', 'Marker', 'o');
fatiguePlotFirst.LineWidth=2;
hold on
eG=errorbar(fatigueData_last_binnedTargets_all{:,3}, fatigueData_last_binnedTargets_all{:,4}, '.');
eG.Color='m';
eG.LineWidth=2;
hold on

% plot borg

% group stats
borgData_last_binnedTargets_sub = grpstats(borgData_last,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
borgData_last_binnedTargets_all = grpstats(borgData_last_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level

% actual plot
borgPlotFirst = plot(borgData_last_binnedTargets_all.binnedTargets, borgData_last_binnedTargets_all{:,3}, 'b', 'Marker', 'o');
borgPlotFirst.LineWidth=2;
hold on
eG=errorbar(borgData_last_binnedTargets_all{:,3}, borgData_last_binnedTargets_all{:,4}, '.');
eG.Color='b';
eG.LineWidth=2;
hold on

% Make the graphs pretty
set(gca, 'FontSize', 12)
hold on
ylabel('Subjective Experience Rating', 'FontSize', 14)
hold on
xlabel('Trial (z-scored)', 'FontSize', 14)
hold on
set(gca, 'xtick', 1:num_bins, 'xticklabel', last_binValues(1:num_bins));
set(gca, 'ylim', [.25, .6])
set(gca,'TickDir','out');

box off;

saveas(gcf, 'raw_ratings_from_isBorgFatigue_split_last.png')

%% Raw plots, ratings from borg vs. fatigue and grip force

num_bins = 5;

% subset borg and fatigue data
fatigueData = borgFatigueTbl((borgFatigueTbl.isBorgFatigue==1),1:width(borgFatigueTbl));
borgData = borgFatigueTbl((borgFatigueTbl.isBorgFatigue==0),1:width(borgFatigueTbl));

% time plots
figure()

binned_var = fatigueData.Time;
[bins, binValues] = discretize(binned_var, num_bins);

fatigueData.binnedTargets = bins;
borgData.binnedTargets = bins;

% group stats -- first half 

figure(1);
% plot fatigue
fatigueData_binnedTargets_sub = grpstats(fatigueData,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
fatigueData_binnedTargets_all = grpstats(fatigueData_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level
% actual plot
fatiguePlotFirst = plot(fatigueData_binnedTargets_all.binnedTargets, fatigueData_binnedTargets_all{:,3}, 'm', 'Marker', 'o');
fatiguePlotFirst.LineWidth=2;
hold on
eG=errorbar(fatigueData_binnedTargets_all{:,3}, fatigueData_binnedTargets_all{:,4}, '.');
eG.Color='m';
eG.LineWidth=2;
hold on

% plot borg

% group stats
borgData_binnedTargets_sub = grpstats(borgData,{'SubID','binnedTargets'},{'mean'},'DataVars',{'BorgFatigueVal'}); % subject and bin level 
borgData_binnedTargets_all = grpstats(borgData_binnedTargets_sub,{'binnedTargets'},{'mean','sem'},'DataVars',{strcat('mean_','BorgFatigueVal')}); % overall level

% actual plot
borgPlotFirst = plot(borgData_binnedTargets_all.binnedTargets, borgData_binnedTargets_all{:,3}, 'b', 'Marker', 'o');
borgPlotFirst.LineWidth=2;
hold on
eG=errorbar(borgData_binnedTargets_all{:,3}, borgData_binnedTargets_all{:,4}, '.');
eG.Color='b';
eG.LineWidth=2;
hold on

% Make the graphs pretty
set(gca, 'FontSize', 12)
hold on
ylabel('Subjective Experience Rating', 'FontSize', 14)
hold on
xlabel('Grip Force (z-scored)', 'FontSize', 14)
hold on
set(gca, 'xtick', 1:num_bins, 'xticklabel', binValues(1:num_bins));
set(gca, 'ylim', [.25, .6])
set(gca,'TickDir','out');

box off;

saveas(gcf, 'raw_ratings_from_isBorgFatigue_gripForce.png')

%% Plot predicted subjective ratings from crossover interaction with time as main covariate
% Create datatable for predict function
dat = borgFatigueTbl;
datNew = table();
% X
datNew.Time = linspace(nanmin(dat.Time), nanmax(dat.Time))';
% covariates of no interest
datNew.Difficulty = (0*ones(1,height(datNew)))';
datNew.Confidence = (nanmean(dat.Confidence)*ones(1,height(datNew)))';
datNew.Effort = (0*ones(1,height(datNew)))';
datNew.cIsBorgFatigue = categorical(0*ones(1,height(datNew))');
datNew.SubID = repmat(dat.SubID(1),height(datNew),1);
% Y
[ypred_Fatigue_marg,yCIpred_Fatigue_marg] = predict(borg_fatigue_model,datNew,...
    'Conditional',false,'DFmethod','satterthwaite');

ySE = yCIpred_Fatigue_marg(:,2) - ypred_Fatigue_marg;

figure();

[y_pred_marg, yCI_marg] = predict(borg_fatigue_model, datNew,...
    'Conditional',false,'DFmethod','satterthwaite');

shadedErrorBar(datNew.Time, y_pred_marg, yCI_marg(:,2)-y_pred_marg, 'lineprops', {'b', 'LineWidth', 3})

% Figure of predicted perceived exertion 
shadedErrorBar(datNew.Time, ypred_Fatigue_marg, ySE,'lineprops',{'b','LineWidth', 3})

hold on

% Figure of predicted fatigue 
datNew.cIsBorgFatigue = categorical(1*ones(1,height(datNew))');

[ypred_Fatigue_marg,yCIpred_Fatigue_marg] = predict(borg_fatigue_model,datNew,...
    'Conditional',false,'DFmethod','satterthwaite');
ySE = yCIpred_Fatigue_marg(:,2) - ypred_Fatigue_marg;

shadedErrorBar(datNew.Time,ypred_Fatigue_marg,ySE,'lineprops',{'m','LineWidth', 3});
hold on

% Make the graphs pretty
set(gca, 'FontSize', 12)
hold on
ylabel('Subjective Experience Rating', 'FontSize', 14)
hold on
xlabel('Trial (z-scored)','FontSize', 14)
set(gca,'TickDir','out');
box off
% hold on

% Save
filename = strcat('predicted_',outcome,'Trial','.png');
saveas(gcf,filename)

% Legend
% hold on
% h = findobj(gca,'Type','line');
% legend([h(1), h(4)],{'Fatigue','Perceived Exertion'})

%% Plot predicted subjective ratings from crossover interaction with grip force as main covariate
datNew.Effort = linspace(nanmin(dat.Effort), nanmax(dat.Effort))';
datNew.Time = (nanmean(dat.Time)*ones(1,height(datNew)))';

figure();

% Figure of predicted perceived exertion 
datNew.cIsBorgFatigue = categorical(0*ones(1,height(datNew))');
[ypred_Fatigue_marg,yCIpred_Fatigue_marg] = predict(borg_fatigue_model,datNew,...
    'Conditional',false,'DFmethod','satterthwaite');
shadedErrorBar(datNew.Effort,ypred_Fatigue_marg,(yCIpred_Fatigue_marg(:,2)-ypred_Fatigue_marg),'lineprops',{'b','LineWidth', 3});
hold on

% Figure of predicted fatigue 
datNew.cIsBorgFatigue = categorical(1*ones(1,height(datNew))');
[ypred_Fatigue_marg,yCIpred_Fatigue_marg] = predict(borg_fatigue_model,datNew,...
    'Conditional',false,'DFmethod','satterthwaite');
shadedErrorBar(datNew.Effort,ypred_Fatigue_marg,(yCIpred_Fatigue_marg(:,2)-ypred_Fatigue_marg),'lineprops',{'m','LineWidth', 3});
hold on

% Make the graphs pretty
set(gca, 'FontSize', 12)
hold on
ylabel('Subjective Experience Rating', 'FontSize', 14)
hold on
xlabel('Grip Force (z-scored)','FontSize', 14)
% Legend
hold on
h = findobj(gca,'Type','line');
lgd = legend([h(1), h(4)],{'Fatigue','Perceived Exertion'}, 'Location', 'northwest');
set(gca,'TickDir','out');
box off
fontsize(lgd,14,'points')
filename = strcat('predicted_',outcome,'GripForce','.png');
saveas(gcf,filename)