% Custom regression layer with Huber loss cost function
%
%  Written by Jari Korhonen, Shenzhen University
%

classdef huberRegressionLayer < nnet.layer.RegressionLayer
    
    methods
        function layer = huberRegressionLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Huber loss regression';
        end

        
        function dLdY = backwardLoss(layer, Y, T)
            % loss = backwardLoss(layer, Y, T) returns the Huber loss 
            % gradients between predictions Y and training targets T.
            
            h=1/9;

            % Calculate Huber loss gradient
            meanAbsoluteError = abs(Y-T);
            maskA = meanAbsoluteError <= h;
            maskB = 1-maskA;
            dLdY = maskA.*(Y-T) + h*maskB.*sign(Y-T);  
        end        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Huber loss 
            % between the predictions Y and the training targets T.
            
            s1 = 1;
            s2 = 2;
            if length(size(Y))>2
                s1 = 3;
                s2 = 4;
            end    
            h=1/9;

            % Calculate Huber loss
            R = size(Y,s1);

            meanAbsoluteError = abs(Y-T);
            maskA = meanAbsoluteError <= h;
            maskB = 1-maskA;
            huberError = 0.5*maskA.*meanAbsoluteError.^2 + ...
                         h*maskB.*(meanAbsoluteError-h/2);          
            huberError = sum(huberError,s1)/R;
    
            % Take mean over mini-batch.
            N = size(Y,s2);
            loss = sum(huberError,s2)/N;
        end
    end
end