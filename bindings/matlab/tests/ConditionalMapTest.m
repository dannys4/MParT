classdef ConditionalMapTest < matlab.unittest.TestCase
    % Copyright 2014 - 2016 The MathWorks, Inc.

    methods ( Test )

        function TestNumCoeffs( testCase )
            [component,~,~] = setup();
            testCase.verifyEqual( component.numCoeffs, uint32(2) );
        end

        function TestCoeffMap( testCase )
            [component,~,~] = setup();
            component.SetCoeffs(zeros(1,component.numCoeffs));
            testCase.verifyEqual( component.CoeffMap(), [0 0] );

            coeffs = randn(1,component.numCoeffs);
            component.SetCoeffs(coeffs);
            testCase.verifyEqual( component.CoeffMap(), coeffs );
        end

        function TestCoeffBounds( testCase )
            [component,~,~] = setup();
            [lb, ub] = component.CoeffBounds();
            testCase.verifyEqual( uint32(numel(lb)), component.numCoeffs );
            testCase.verifyEqual( uint32(numel(ub)), component.numCoeffs );

            testCase.verifyEqual( max(lb), -Inf );
            testCase.verifyEqual( min(ub), Inf );
        end

        function TestEvaluate( testCase )
            [component,num_samples,x] = setup();
            testCase.verifyEqual( size(component.Evaluate(x)), [1,num_samples] );
        end

        function TestLogDeterminant( testCase )
            [component,num_samples,x] = setup();
            testCase.verifyEqual( size(component.LogDeterminant(x)), [num_samples,1] );
        end

        function TestInverse( testCase )
            [component,num_samples,x] = setup();
            coeffs = randn(component.numCoeffs,1);
            component.SetCoeffs(coeffs);
            y = component.Evaluate(x);
            x_ = component.Inverse(zeros(1,num_samples),y);
            testCase.verifyTrue( all( abs(x_ - x(end,:)) < 1E-3 ) );
        end


    end
end

function [component, num_samples, x] = setup()
    opts = MapOptions();

    multis = [0;1];  % linear
    mset = MultiIndexSet(multis).Fix();

    component = CreateComponent(mset, opts);
    num_samples = 100;
    x = randn(1,num_samples);
end