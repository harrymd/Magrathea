function [] = run_gravity(dir_model, name_model, pOrder, anomaly_str)
    %% Calculates the gravity field of a planetary model.
    % Adapted from https://github.com/js1019/PlanetaryModels.
    
    %dir_model = '/Users/hrmd_work/Documents/research/stoneley/output/Magrathea/prem_0473.4_300/';
    %name_model = 'model';
    
    fprintf('Calculating gravity\n');
    tic
    
    % Include FMM library.
    addpath('fmmlib3d-1.2/matlab/');
    
    % Include Nodal Discontinuous Galerkin codes (taken from
    % https://github.com/tcew/nodal-dg).
    addpath('nodal-dg/');

    % Identify planetary model directory.
    fmesh = fullfile(dir_model, name_model);

    % Probably delete this
    % Load some useful variables.
    %load ../radialmodels/prem3L_noocean_gravity.mat

    % Probably delete this
    %saveFMM = true;

    % Finite element order (choose 1 or 2)
    %pOrder  = 2;

    % Scaling.
    %scaling = 6.371*10^3;

    % Load mesh.
    % [n_pts]           Number of mesh points.
    % [n_tet]           Number of tetrahedra.
    % pout  (n_pts, 3)  Points      
    % tout  (n_tet, 4)  Tetrahedra.
    % at    (n_tet, 1)  Attributes.
    [pout,tout,~,at,~] = read_mesh3d([fmesh,'.1']);

    % Identify file names.
    fmid    = ['_pod_',int2str(pOrder),'_'];
    fname = [fmesh,'.1']; 
    ftail   = ['true_', anomaly_str, '.dat'];
    frho    = [fname,'_rho',fmid,ftail];
    fgfld = [fname,fmid,'potential_acceleration_',ftail];
    %fvtk = [fname,fmid,'gravity.vtk'];
    
    % Identify precision of TetGen files. 
    accry = 'float64';
    
    % Set gravitational constant.
    G = 6.6723*10^-5;

    % Number of elements.
    Ne = size(tout,1);
    
    % Number of points per tetrahedron (4 for pOrder = 1, 10 for pOrder =
    % 2)
    pNp = (pOrder+1)*(pOrder+2)*(pOrder+3)/6;

    % Read density model.
    % rho   (n_tet*4, 1) Density at each tetrahedron vertex (flattened).
    fid =fopen(frho);
    rho = fread(fid,Ne*pNp,accry);
    fclose(fid);

    % Unflatten density array.
    % rho (n_pts_per_tet, n_tet) Density at each tetrahedron vertex.
    rho0 = reshape(rho,pNp,Ne);

    % Compute the detJ.
    N = pOrder; 
    
    % Get the coordinates of a reference equilateral tetrahedron.
    [x,y,z] = Nodes3D(N); 
    
    % Transfer to generalised tetrahedral coordinates.
    [r,s,t] = xyztorst(x,y,z);
    
    % Calculate the Vandermonde matrix.
    V = Vandermonde3D(N,r,s,t);
    
    % Calculate the D matrix.
    [Dr,Ds,Dt] = Dmatrices3D(N, r, s, t, V);
    
    % Calculate the mass matrix.
    Mass = inv(V)'*inv(V);
    
    % Calculate the nodal coordinates (coordinates of each point in 
    % each tetrahedron).
    va = tout(:,1)';
    vb = tout(:,2)';
    vc = tout(:,3)';
    vd = tout(:,4)';
    x = 0.5*(-(1+r+s+t)*pout(va,1)'+(1+r)*pout(vb,1)'+(1+s)*pout(vc,1)'+(1+t)*pout(vd,1)');
    y = 0.5*(-(1+r+s+t)*pout(va,2)'+(1+r)*pout(vb,2)'+(1+s)*pout(vc,2)'+(1+t)*pout(vd,2)');
    z = 0.5*(-(1+r+s+t)*pout(va,3)'+(1+r)*pout(vb,3)'+(1+s)*pout(vc,3)'+(1+t)*pout(vd,3)');
    
    % Get the Jacobian.
    [~,~,~,~,~,~,~,~,~,J] = GeometricFactors3D(x,y,z,Dr,Ds,Dt);
  
    % Get the links between nodes.
    [~,~,~,tet] = construct(fname,pOrder);

    tnew = reshape(1:size(tet,1)*size(tet,2),size(tet,2),size(tet,1));
    psiz = max(tet(:));
    pnew0 = zeros(psiz,3);
    pnew0(:,1) = x(:); 
    pnew0(:,2) = y(:); 
    pnew0(:,3) = z(:);

    pnew = pnew0(tet',:);

    xd = reshape(pnew0(tet',1),4,size(tet,1)); 
    yd = reshape(pnew0(tet',2),4,size(tet,1)); 
    zd = reshape(pnew0(tet',3),4,size(tet,1)); 


    % Prepare the sources
    iprec= 5; %5; 
    Nenew = size(tet,1);
    nsource = Nenew; 

    source(1,:) = ones(1,4)*xd/4;
    source(2,:) = ones(1,4)*yd/4;
    source(3,:) = ones(1,4)*zd/4;

    ifcharge = 1; 
    charge0 = J(1,:).*sum(Mass*rho0)/pOrder^3;
    charge = reshape(ones(pOrder^3,1)*charge0,1,Nenew);

    ifdipole = 0;
    dipstr = zeros(1,Nenew);
    dipvec = rand(3,Nenew);
    ifpot = 0;
    iffld = 0;
    %ntarget = Ne*pNp; 
    %target(1,:) = reshape(x,ntarget,1);
    %target(2,:) = reshape(y,ntarget,1);
    %target(3,:) = reshape(z,ntarget,1);

    %ntarget = size(pout,1); 
    %target = pout'; 
    ntarget = size(pnew0,1); 
    target = pnew0'; 

    ifpottarg = 1;
    iffldtarg = 1;
     
    if (nsource+ntarget < 20e6)

        % Use FMM.
        [U]=lfmm3dpart(iprec,nsource,source,ifcharge,charge,...
            ifdipole,dipstr,dipvec,ifpot,iffld,ntarget,target,ifpottarg,iffldtarg);

        rnrm = sqrt(sum(target.*target));
        gnrm = sqrt(sum(real(U.fldtarg).*real(U.fldtarg)))*G;

        %max(gnrm(:))
        %min(real(U.pottarg(:))*G)
        %U.ier

        gfld0 = - real(U.fldtarg(:,:)')*G;

        % semi-analytic solutions
        %gnrm0 = interp1(RI,gref*G,rnrm,'pchip');
    end
    %rnrm1 = sqrt(sum(pnew0'.*pnew0')); 
    %gnrm1 = interp1(RI,gref*G,rnrm1,'pchip');

    %plot(rnrm,gnrm0,'o'); hold on;
    %plot(rnrm,gnrm,'+')


    %gfld1 = zeros(size(pnew0));
    %for i = 1:3
       %gfld1(:,i) = - pnew0(:,i)./rnrm1(:).*gnrm1(:); 
    %end

    %gfld1t = gfld1';

    %if (pOrder==1 && nsource+ntarget < 20e6)
    %resg = gfld1 - gfld0;
    %norm(resg(:))/norm(gfld1(:))
    %end 

    saveFMM = true;
    if saveFMM % FMM
    % save the data
    fprintf('Writing to %s\n', fgfld);
    fid=fopen(fgfld,'w');
    fwrite(fid,gfld0',accry);
    fclose(fid);
    %else % semi-analytic
    %% save the data
    %fid=fopen(fgfld,'w');
    %fwrite(fid,gfld1',accry); % first 3 directions; pNp, ntet
    %fclose(fid);
    end 

    saveVTK = 0;
    if (saveVTK)
    gfld = -real(U.fldtarg(:,tet'))*G;
    %gfld = -real(U.fldtarg(:,tout'))*G;
    gpot = -real(U.pottarg(:))*G;
    % visual
    filename = fvtk;
    data_title = 'Gravity';
    % organize data
    % .type field must be 'scalar' or 'vector'
    % number of vector components must equal dimension of the mesh
    data_struct(1).type = 'scalar';
    data_struct(1).name = 'potential';
    data_struct(1).data = -real(U.pottarg(:))*G;

    data_struct(2).type = 'vector';
    data_struct(2).name = 'field';
    data_struct(2).data = -real(U.fldtarg(:))*G;
    flipped = false;

    % otherwise, if you want to transpose the data, then set this to *true*
    %tnew = reshape(1:size(tout,1)*size(tout,2),size(tout,2),size(tout,1));
    %pnew = pout(tout',:)/scaling;  
    % write the file

    stat = vtk_write_tetrahedral_grid_and_data(filename,data_title,pnew0/scaling,...
        tet,data_struct,flipped);
    %toc
    end

    %filename = [fname,fmid,'trueG.vtk'];
    %data_title = 'Gravity';
    %data_struct(1).type = 'vector';
    %data_struct(1).name = 'field';
    %data_struct(1).data = gfld1t(:);
    %flipped = false;

    %stat = vtk_write_tetrahedral_grid_and_data(filename,data_title,pnew0/scaling,...
    %    tet,data_struct,flipped);
    %toc
    toc
end