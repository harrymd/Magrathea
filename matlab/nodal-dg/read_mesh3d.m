function [p,t,pt,at,neigh] = read_mesh3d(fname_base)
    %% Reads the .ele, .node and .neigh files from TetGen.
    % Input:
    %
    % fname_base    Path of the .ele, .node and .neigh files without the
    %               file suffixes.
    %
    % Output:
    %
    % [n_pts]           Number of mesh points.
    % [n_tet]           Number of tetrahedra.
    % p     (n_pts, 3)  Mesh points.      
    % t     (n_tet, 4)  Tetrahedra indices.
    % pt    (n_tet, 3)  Coordinates of centres of tetrahedra.
    % at    (n_tet)     Tetrahedra attributes.
    % neigh (n_tet, 4)  Indices of neighbours of each tetrahedron.
    
    %% Read the .ele file.
    fname1 = fname_base;
    fid = fopen([fname1 '.ele'],'r');
    
    % Read the header line.
    A=fscanf(fid,'%d %d %d',3);
    Nele = A(1); % Number of tetrahedron.
    Nattr = A(3); % Number of attributes.
    A = fscanf(fid,'%d',Nele*(5+Nattr));
    A = reshape(A,5+Nattr,Nele)';
    t=A(:,[2 3 4 5]); % tri ID, node 1~3
    fclose(fid);
    if(Nattr>0)
        at=A(:,6);
    else
        at=[];
    end
    fid = fopen([fname1 '.node'],'r'); %read in .node file
    A=fscanf(fid,'%d',4); 
    Nnodes = A(1); % Number of nodes
    Nattr = A(3); % Number of attributes
    Nbdry = A(4); % Number of boundaries
    A=fscanf(fid,'%g',Nnodes*(4+Nattr+Nbdry));
    A = reshape(A,4+Nattr+Nbdry,Nnodes)';
    p = A(:,[2 3 4]);
    pt=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:)+p(t(:,4),:))/4;
    
    fname1=[fname_base,'.neigh'];
    if(exist(fname1,'file'))
        fid=fopen(fname1,'r'); %read in .neigh file
        fscanf(fid,'%d',2);
        A = fscanf(fid,'%d',Nele*5);
        A = reshape(A,5,Nele)';
        neigh=A(:,2:5);
    else
        neigh=[];
    end
    
end