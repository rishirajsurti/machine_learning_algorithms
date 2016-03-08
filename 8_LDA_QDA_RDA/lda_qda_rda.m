%%
%run and initialize initPmtk3.m

requireStatsToolbox(); 

load fisheriris

if 0
  figure
gscatter(meas(:,3), meas(:,4), species,'rgb','osd');
xlabel('Petal length');
ylabel('Petal width');
end
N = size(meas,1);

s = RandStream('mt19937ar','seed',0);
RandStream.setDefaultStream(s);

types  ={'linear', 'quadratic', 'diagQuadratic'};
for t=1:3
  type = types{t};
  

ldaClass = classify(meas(:,3:4),meas(:,3:4),species, type);
[ldaResubCM,grpOrder] = confusionmat(species,ldaClass);
bad = ~strcmp(ldaClass,species);
ldaResubErr = sum(bad) / N

cp = cvpartition(species,'k',10);
ldaClassFun= @(xtrain,ytrain,xtest)(classify(xtest,xtrain,ytrain, type));
ldaCVErr  = crossval('mcr',meas(:,3:4),species,'predfun', ...
             ldaClassFun,'partition',cp)
           
% Plot data and errors
figure;
gscatter(meas(:,3), meas(:,4), species,'rgb','osd');
xlabel('Petal length');
ylabel('Petal width');
hold on;
%plot(meas(bad,3), meas(bad,4), 'kx', 'markersize', 10, 'linewidth', 2);
title(sprintf('%s, train error %5.3f, cv error %5.3f',type,  ldaResubErr, ldaCVErr))
%title(sprintf('Iris Dataset: Petal Length & Petal Width'))

% Plot decision boundary
figure;
[x,y] = meshgrid(1:.1:7,0:.1:2.5);
x = x(:);
y = y(:);
j = classify([x y],meas(:,3:4),species, type);
gscatter(x,y,j,'grb','sod')
xlabel('Petal length');
ylabel('Petal width');
title(sprintf('%s, train error %5.3f, cv error %5.3f', type, ldaResubErr, ldaCVErr))
end
