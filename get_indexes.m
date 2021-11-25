function [nu_vec delta_vec sn_vec kc] = get_indexes(nu,delta,k1,k2)

% Add path for both Yalmip and Mosek directories
  %addpath(genpath('YALMIP LOCATION'))
  %addpath(genpath('MOSEK LOCATION'))

% Nu and Delta are the network parameters
% K1 and K2 are the sector constraints of each nonlinear activation function.
% nu_vec and delta_vec return the individual parameters for each layer
% sn_vec contains the spectral bound for each layer
% kc is the sector bound in which the network is contained. That is
% kc1.x <= y <= kc2.x

l=length(k1);
nu_vec_var = sdpvar(1,l);
delta_vec_var = sdpvar(1,l);

kc1 = (1-sqrt(1-4*value(delta)*value(nu)))/(2*value(delta));
kc2 = (1+sqrt(1-4*value(delta)*value(nu)))/(2*value(delta));
kc = [kc1 kc2];

index_matrix = diag([nu_vec_var -delta]+[-nu delta_vec_var])- diag(0.5*ones(l,1),-1) -diag(0.5*ones(l,1),1) ;
index_matrix(1,size(index_matrix,2))=+0.5;
index_matrix(size(index_matrix,1),1)=+0.5;

opt = sdpsettings;
opt.verbose =0;

obj = delta_vec_var(l);
%obj = norm(nu_vec_var)+norm(delta_vec_var);
optimize([index_matrix>=0 delta_vec_var>=0 nu_vec_var<=0],obj,opt);
optimize([index_matrix>=0 delta>=0 nu_vec_var<=0 delta_vec_var>=0 abs(nu_vec_var(l))<=0.25/abs(value(delta_vec_var(l))) abs(nu_vec_var(l))>=0.125/abs(value(delta_vec_var(l)))],obj,opt);
nu_vec_var(l) = -0.2499/abs(value(delta_vec_var(l)));
delta_vec_var(l)= value(delta_vec_var(l));
obj = trace(index_matrix);
optimize([index_matrix>=0 delta>=0 nu_vec_var<=0 delta_vec_var>=0],obj,opt);
%nu_vec_var<=0
nu_vec = value(nu_vec_var);
delta_vec = value(delta_vec_var);

m = (k1+k2)/2;
p = k1.*k2;
%sn_vec = (-m+sqrt((8*m.^2-16*p).*abs(nu_vec.*delta_vec)+2*p))./(2*abs(delta_vec).*(m.^2-2*p));

%D = (m.^2).*(abs(delta_vec)-p.*abs(nu_vec)).^2-4*p.*(abs(delta_vec)-0.5).*(0.5-p.*abs(nu_vec));
%sn_vec = (m.*(p.*abs(nu_vec)+abs(delta_vec))+sqrt(D))./(2*p.*(abs(delta_vec-0.5)));

D = (1./m).*((1-p).*(abs(delta_vec.*nu_vec))./(abs(delta_vec)-abs(nu_vec))-0.5);
sn_vec = D./(m.*D+abs(delta_vec)+0.5);

%D = m.^2.*(abs(delta_vec)+abs(nu_vec)).^2-4.*p.*(0.5+abs(delta_vec).*(0.5-abs(nu_vec)));
%sn_vj = (-m.*(abs(nu_vec)+abs(delta_vec))+sqrt(D))./(2*p.*(0.5+abs(delta)));

end