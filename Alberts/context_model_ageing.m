function [logL] = context_model_controls(params, Stim, Resp, subj)
% [logL] = context_model(params, Stim, Resp, model_option) takes the free
% parameters of the context model into account and fits the model to the
% data collected from subjects and stored in a stimulus vector Stim and
% response vector Resp. The fit with the lowest log likelihood value will
% be the best fit, hence this function is minimized using fminsearch or
% fmincon. 

sigma_prior = params(1); % prior knowledge variance
a0 = params(2); % offset otoliths
a1 = params(3); % otolith noise increase with tilt angle
Aocr = params(4); % uncompensated ocular counterroll
K1 = params(5); % vertical contextual prior
K2 = params(6); % horizontal contextual prior
num = params(7);
if length(params) > 7;
    P_lapse = params(8);
else P_lapse = 0;
end

% Other variables
tilt_angle = [0,30]; % head tilt
tilt = -180:0.1:180; % tilt range
frame_line = -45:5:40; % frame orienations
angle = [0 90 180 270]; % seperation of the different contextual priors
logL = 0;

% 
% sig_SVV_est = zeros(length(tilt_angle),length(frame_line));
% mu = zeros(length(tilt_angle),length(frame_line));
% mu_SVV_est = zeros(length(tilt_angle),length(frame_line));

for j = 1:length(tilt_angle)% we only run the frame conditions here 
 

    % Otolith probability distribution
    prob_otol = normpdf(tilt, tilt_angle(j), (a0+a1*abs(tilt_angle(j))));
    prob_otol = prob_otol/trapz(tilt*pi/180,prob_otol);
    
    % Prior probability distribution
    prob_prior = normpdf(tilt, 0, sigma_prior);
    prob_prior=prob_prior/trapz(tilt*pi/180,prob_prior);
    
    % Prior contextual distribuation
    for i = 1:length(frame_line)
%         if isnan(angles(i,length(sub)*(j-1)+subj)) == 0 
        frameline = -(frame_line(i)-tilt_angle(j))-Aocr*sin(tilt_angle(j)*pi/180);
        if frameline > 45
           frameline = frameline - 90;
        elseif frameline < -45;
           frameline = frameline + 90;
        end
       %frameline          
       K = [K1-((1-cosd(abs(2*frameline))).*num).*(K1-K2) K2+((1-cosd(abs(2*frameline))).*(1-num)).*(K1-K2)...
              K1-((1-cosd(abs(2*frameline))).*num).*(K1-K2) K2+((1-cosd(abs(2*frameline))).*(1-num)).*(K1-K2)]; % Calculate the different K's      
           
        prob_context_ad = zeros(size(tilt));
        for m = 1:length(angle)
            prob_context_ad = prob_context_ad + exp(K(m)*cosd((frameline+angle(m))-tilt))/(2*pi*besseli(0,K(m)));
%             prob_context_ad = prob_context_ad + circ_vmpdf(frameline+angle(m),tilt,K(m));
        end
        prob_context_ad = prob_context_ad./length(angle);
        prob_context_ad = prob_context_ad/trapz(tilt*pi/180,prob_context_ad);

        % Calculate and characterize the posterior
        p = prob_otol.*prob_prior.*prob_context_ad;
        cumd = cumtrapz(p./trapz(p)); % cumulative density function of the posterior
        E_s_cumd = -tilt+tilt_angle(j)-Aocr*sin(tilt_angle(j)*pi/180);
        
        if sum(~isnan(cumd))<2
            logL = 10000;
        else    
        spline_coefs=spline(E_s_cumd,cumd);
        P_left{i}=0.5*P_lapse+(1-P_lapse)*ppval(spline_coefs,Stim{subj,j,i});
        Pvec_SVV{i} = max(eps,P_left{i}.*(1-Resp{subj,j,i})+(1-P_left{i}).*Resp{subj,j,i});
        
        logL = logL+sum(-log(Pvec_SVV{i}));
        end
%         else logL = logL + 0;
%         end
    end
end

%% also run the darkSVV 30 deg tilt condition

% Otolith probability distribution
tilt_angle = 30;
prob_otol = normpdf(tilt, tilt_angle, (a0+a1*abs(tilt_angle)));
prob_otol = prob_otol/trapz(tilt*pi/180,prob_otol);
    
% Prior probability distribution
prob_prior = normpdf(tilt, 0, sigma_prior);
prob_prior=prob_prior/trapz(tilt*pi/180,prob_prior);

% Calculate and characterize the posterior
p = prob_otol.*prob_prior;
cumd = cumtrapz(p./trapz(p)); % cumulative density function of the posterior
E_s_cumd = -tilt+tilt_angle-Aocr*sin(tilt_angle*pi/180);

spline_coefs=spline(E_s_cumd,cumd);
P_left=0.5*P_lapse+(1-P_lapse)*ppval(spline_coefs,Stim{subj,3,1});
Pvec_SVV = max(eps,P_left.*(1-Resp{subj,3,1})+(1-P_left).*Resp{subj,3,1});

logL = logL+sum(-log(Pvec_SVV));

        