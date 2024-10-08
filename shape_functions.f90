
        program mains ! ann filter and simple code
!        if(.true.) then ! Sn quadrature on unit sphere
        if(.false.) then ! Sn quadrature on unit sphere
           call file_generator4octant_sn_quadrature
        else if(.false.) then ! voronoi grid to triangles
           call main32
        else if(.false.) then! solver example...
           call main7
        else if(.true.) then! solver example...
!         call cty_filter_orig ! linear cty element filters
! all the filters: linear, quadratic etc cty filters
           call set_up_cty_filter_quadratic_and_higher_fem
!           call main4 ! linear DG element filters
        endif
        stop 2992
        end program mains
! 
! 
! 
! 
        subroutine main32 ! voronoi grid to triangles testing
        implicit none
! integers representing the length of arrays...
! nx=no of nodes across, ny=no of nodes up. 
!        integer, parameter :: nx=11,ny=11, nonods=nx*ny, totele=(nx-1)*(ny-1)
        integer, parameter :: nx=128,ny=128, max_voronoi_cells=10000, max_connect=70
        integer, parameter :: nx2=2*nx,ny2=2*ny
! mesh_functional: This subdoutine calculates the quality if the triangle mesh mesh_functional from 
! a grid gof pixes of different color defined by color(i,j)=colour of pixel (integer) and 
! the grid or image has dimensions (nx,ny) - number of cells across and down respectively. 
! xc,yc: It calculates the centres of mass of the Voronoi cells (xc(icell),yc(icell)) in which icell 
! is the cell number. 
! number_voronoi_cells: It calculates the number of Voronoi cells number_voronoi_cells. 
! dx, dy: are the dimensions of the cells. 
! desired_edge_length is the desired edge length of the triangles. 
! max_voronoi_cells contains the maximum number of Voronoi cells in the image. 
! max_connect is the maximum number of Voronoi cells surrounding (next too) a given Voronoi cell. 
! voronoi_cell_connect(icell, jj)= the jj^th index to a aurrounding Voronoi cell of cell icell. 
! voronoi_cell_connect(icell, jj) contains a non-zero only up to the number jj of surrounding 
! Voronoi cells otherwise it contains 0. 
          real xc(max_voronoi_cells), yc(max_voronoi_cells)
          real mesh_functional
          integer voronoi_cell_connect(max_voronoi_cells,max_connect)
          integer number_voronoi_cells
          real dx, dy, desired_edge_length
          integer color(nx,ny),color2(2*nx,2*ny)
! Local variables
          integer i,j,ii,jj,nnx,nny,icolor,ivor, imax, iconnect

          dx=1.0
          dy=1.0
          desired_edge_length=4.0
          nnx=4
          nny=4
          color=0 
          do jj=1,nny
          do ii=1,nnx
             icolor=ii + (jj-1)*nnx
             do j=1,ny
             do i=1,nx  
                if(1 + (i-1)/(nx/nnx) == ii) then
                if(1 + (j-1)/(ny/nny) == jj) then
                   color(i,j)=icolor
                endif 
                endif 
             end do
             end do
          end do
          end do
! 
        open(3, file='example.txt')
        do j=1,ny
           read(3,*) (color(i,j),i=1,nx)
        end do
        close(3)
        imax=maxval(color)
          do j=1,ny
          do i=1,nx 
             if(color(i,j)==0) color(i,j)=imax 
          end do
          end do

!          if(minval(color)==0) then
!             stop 292
!          endif
! 
          do j=1,ny
             print *,'j,color(:,j):',j,':',color(:,j)
          end do
          do j=1,ny
          do i=1,nx 
             do jj=1,2
             do ii=1,2
                color2((i-1)*2+ii,(j-1)*2+jj) = color(i,j)
             end do
             end do
          end do
          end do
! 
          call voronoi_cell_functional_and_pts(xc,yc, mesh_functional, &
                                             voronoi_cell_connect, number_voronoi_cells, &
                                             color2, dx, dy, desired_edge_length, &
                                             nx2,ny2, max_voronoi_cells, max_connect ) 
!                                             color, dx, dy, desired_edge_length, &
!                                             nx,ny, max_voronoi_cells, max_connect ) 
          print *,'just outside voronoi_cell_functional_and_pts'
! 
! 
       if(.true.) then
          open(3, file='mesh_data.txt')
          write(3,*) mesh_functional
          write(3,*) number_voronoi_cells
          do ivor=1,number_voronoi_cells
             write(3,*) ivor,xc(ivor),yc(ivor)
             imax=0
             do iconnect=max_connect,1,-1
                if(voronoi_cell_connect(ivor,iconnect)==0) imax=iconnect
             end do
             write(3,*) imax
             write(3,*) (voronoi_cell_connect(ivor,iconnect),iconnect=1,imax-1)
          end do
       endif
       if(.false.) then
          do ivor=1,number_voronoi_cells
             print *,'ivor,xc(ivor),yc(ivor):',ivor,xc(ivor),yc(ivor)
             imax=0
             do iconnect=max_connect,1,-1
                if(voronoi_cell_connect(ivor,iconnect)==0) imax=iconnect
             end do
             print *,'voronoi_cell_connect(ivor,:):', &
                 (voronoi_cell_connect(ivor,iconnect),iconnect=1,imax-1)
          end do
        endif
        stop 32
        end subroutine main32
! 
! 
! 
! 
!        programsubroutine main ! ann filter and simple code
        subroutine main7 ! ann filter and simple code
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
! nx=no of nodes across, ny=no of nodes up. 
!        integer, parameter :: nx=11,ny=11, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=1025,ny=1025, nonods=nx*ny, totele=(nx-1)*(ny-1)
        integer, parameter :: nx=513,ny=513, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=257,ny=257, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=129,ny=129, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=5,ny=5, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=65,ny=65, nonods=nx*ny, totele=(nx-1)*(ny-1)
!        integer, parameter :: nx=33,ny=33, nonods=nx*ny, totele=(nx-1)*(ny-1)
        integer, parameter :: nloc=4,sngi=3, ngi=9, ndim=2, &
                              nface=4, max_face_list_no=2
        integer, parameter :: ntime = 10
        real, parameter :: dt=0.1 
! example of node ordering...
!      7-----8-----9
!      !     !     !
!      !     !     !        
!      4-----5-----6
!      !     !     !
!      !     !     !        
!      1-----2-----3
        real b(nonods), ml(nonods)
        real k(nonods), sig(nonods), s(nonods), t_new(nonods), t_old(nonods)
        real u(ndim,nonods)
        real x_all(ndim,nloc*totele)
        integer fina(nonods+1) 
        integer ndglno(nloc*totele)
! filters for ann:
        integer, allocatable :: cola(:), cola_no_bc(:), fina_no_bc(:)
        integer, allocatable :: ion_boundary(:), map2_no_bc_nodes(:), fin_sfc_nonods(:)
        integer, allocatable :: ncurve_whichd(:,:), ncurve_space_fill_curve_numbering(:,:)
        integer, allocatable :: super_sfc_node_ordering_inverse(:)
        integer, allocatable :: full_super_sfc_node_ordering_inverse(:)
        integer, allocatable :: fina_sfc_all_un(:,:), cola_sfc_all_un(:,:)
        integer, allocatable :: ncola_sfc_all_un(:)
        integer, allocatable :: add_ncurve_space_fill_curve_numbering(:,:)
        real, allocatable :: a(:), a_no_bc(:), b_no_bc(:), a_filter(:), relax(:), relax_down(:)
        real, allocatable :: ml_no_bc(:), psi(:), psi_no_bc(:), psi_no_bc_guess(:)
        real, allocatable :: a_sfc(:,:,:), b_sfc(:,:), ml_sfc(:,:), max_psi_no_bc(:)
        real, allocatable :: a_sfc_all_un(:,:)
        integer npoly,ele_type, ncola
        integer i,j, count
        integer its,nits, ncurve, graph_trim, iuse_starting_node, icurve, nonods_no_bc
        integer max_nlevel, max_nonods_sfc_all_grids, nlevel, inod_no_bc, inod, inod_new
        integer inod_sfc_all, ncola_no_bc, nonods_sfc_all_grids
        integer iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer nfilt_size_sfc, max_ncola_sfc_all_un
        integer full_super_nonods, super_nonods, ndim_hilbert, max_add_ncurve
        integer n_add_ncurve, add_sfc, vcycle, fcycle, top_level, ilevel
        integer isweep, istart, ifinish, istep
        real relax_keep_off
        real dx,dy, xstart,xfinish,xmid, sx, xpt,ypt, rr
        logical optimal_relax
! 
        ncola=9*nonods ! this is the maximum size of ncola it will be smaller. 
        allocate(a(ncola),a_filter(ncola))
        allocate(cola(ncola))
! 
        print *,'here in main'
!        stop 23
        dx=1.0/real(nx-1)
        dy=1.0/real(ny-1)
! 
        call set_up_grid_problem(t_old, t_new, x_all, ndglno, fina, cola, &
                                 dx, dy, ncola, nx, ny, nloc, ndim)

        npoly=1
        ele_type=1
!        k=0.0; sig=0.0; s=0.0; u(1,:)=1.0; u(2,:)=1.0
        k=1.0; sig=0.0; s=0.0; u(1,:)=0.0; u(2,:)=0.0

        s=0.0
        xstart  = 0.5 - 0.5/8.0
        xfinish = 0.5 + 0.5/8.0
        xmid=0.5
        do j=2,ny-1 ! miss out boundary nodes.
        do i=2,nx-1
           inod=(j-1)*nx+i
           xpt = real(i-1)*dx + 0.0*dx
           ypt = real(j-1)*dy + 0.0*dy
! scale in x-direction
           if((xpt<xstart).or.(xpt>xfinish)) then
              sx=0.0
           else
              if(xpt<xmid) then
                 sx=(xpt-xstart)/(xmid-xstart)
              else
                 sx=(xfinish-xpt)/(xfinish-xmid)
              endif
           endif
! scale in y-direction
           if((ypt<xstart).or.(ypt>xfinish)) then
              s(inod)=0.0
           else
              if(ypt<xmid) then
                 s(inod)=sx*(ypt-xstart)/(xmid-xstart)
              else
                 s(inod)=sx*(xfinish-ypt)/(xfinish-xmid)
              endif
           endif
        end do
        end do
!        s(nonods/2)=1.0
! in python:
! a, b = u2r.get_fe_matrix_eqn(x_all, u, k, sig, s, fina,cola, ncola, ndglno, nonods,totele,nloc, ndim, ele_type)
! ml = the lumped mass 
        call get_fe_matrix_eqn(a,b, ml, k, sig, s, u, x_all, &
                               fina,cola, ndglno,  &
                               ele_type, ndim, totele,nloc, ncola,nonods) 

! Remove all the boundary nodes.
        allocate(ion_boundary(nonods))
        ion_boundary=1
        nonods_no_bc=0
        do j=2,ny-1 ! miss out boundary nodes.
        do i=2,nx-1
           inod=(j-1)*nx+i
           ion_boundary(inod)=0
           nonods_no_bc=nonods_no_bc+1
        end do
        end do
        print *,'1-nx,ny,nonods,nonods_no_bc:',nx,ny,nonods,nonods_no_bc
! 
        allocate(map2_no_bc_nodes(nonods))
        allocate(b_no_bc(nonods_no_bc),ml_no_bc(nonods_no_bc)) 
        allocate(a_no_bc(ncola))
        allocate(fina_no_bc(nonods_no_bc+1),cola_no_bc(ncola))
        call remove_bc_nodes_a_b(a_no_bc, b_no_bc, ml_no_bc, fina_no_bc, cola_no_bc, ncola_no_bc, &
                   map2_no_bc_nodes, a, b, ml, fina, cola,ion_boundary, nonods_no_bc, ncola, nonods)
! 
!        call testing2331
      if(.false.) then
        print *,'nonods,nonods_no_bc:',nonods,nonods_no_bc
        print *,'fina_no_bc:',fina_no_bc
        do inod_no_bc=1,nonods_no_bc
           print *,'inod_no_bc,nonods_no_bc:',inod_no_bc,nonods_no_bc
           do count = fina_no_bc(inod_no_bc), fina_no_bc(inod_no_bc+1)-1 
              print *,'count, cola_no_bc(count), a_no_bc(count):', &
                       count, cola_no_bc(count), a_no_bc(count)
           end do
        end do
      endif
!        nfilt_size_sfc=27
!        nfilt_size_sfc=129!9
        nfilt_size_sfc=0 ! =0 for sparse matrix storage on each multi-grid level (no approx of multi-grid matrix)
! form 2 SFCs...
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.75 works, =0.0 works
        relax_keep_off=0.8 ! works for hard problem =0.9 not work, =0.8 not work, =0.75 works, =0.0 works
!        relax_keep_off=0.5
!        relax_keep_off=0.85
!        relax_keep_off=0.999
!        relax_keep_off=0.75
! need less relaxation the greater the filter width of the sfc. 
        if(nfilt_size_sfc==1) relax_keep_off=0.6
        if(nfilt_size_sfc==3) relax_keep_off=0.8
        if(nfilt_size_sfc==5) relax_keep_off=0.93
        if(nfilt_size_sfc==7) relax_keep_off=0.97
        if(nfilt_size_sfc.ge.9) relax_keep_off=0.999
!        if(nfilt_size_sfc.ge.25) relax_keep_off=0.97
!        if(nfilt_size_sfc.ge.41) relax_keep_off=0.999
!        relax_keep_off=0.999
!        relax_keep_off=0.9
!        ncurve = 2
!        ncurve = 10
        add_sfc = 0 ! add some more sfcs to form matrix.
        ncurve = 1
        graph_trim = -10
        if(add_sfc.ne.0) graph_trim = 4 ! seems the best option for using SFC as matrices
!        if(nfilt_size_sfc==0) graph_trim = 5 ! seems the best option for using SFC as matrices
!        iuse_starting_node = 0 ! ie no starting node
        iuse_starting_node = -1 ! use the rapid within subroutine mehtod
        allocate( ncurve_whichd(nonods_no_bc,ncurve) )
        allocate( ncurve_space_fill_curve_numbering(nonods_no_bc,ncurve) )
        print *,'going into ncurve_python_subdomain_space_filling_curve'
!        call ncurve_python_subdomain_space_filling_curve( ncurve_whichd,  &
        call ncurve_python_subdomain_space_filling_curve_stub( ncurve_whichd,  &
              ncurve_space_fill_curve_numbering,  cola_no_bc,fina_no_bc, iuse_starting_node, &
              graph_trim, ncurve, nonods_no_bc,ncola_no_bc)
        print *,'just out of ncurve_python_subdomain_space_filling_curve'
      if(add_sfc.ne.0) then
        max_add_ncurve=12
        allocate( add_ncurve_space_fill_curve_numbering(nonods_no_bc,max_add_ncurve) )
        call add_sfc_for_matrix(add_ncurve_space_fill_curve_numbering, &
              n_add_ncurve, &
              ncurve_space_fill_curve_numbering,  cola_no_bc,fina_no_bc, &
              ncurve, nonods_no_bc,ncola_no_bc, max_add_ncurve)
      endif ! if(add_sfc.ne.0) then
! 
        if(.false.) then ! test SFC ordering...
           print *,'here******'
           allocate(super_sfc_node_ordering_inverse(2*ncurve*nonods_no_bc))
           call combine_sfcs(super_sfc_node_ordering_inverse, super_nonods, &
                         ncurve_space_fill_curve_numbering,nonods_no_bc,ncurve)
           print *,'super_nonods=',super_nonods
           print *,'super_sfc_node_ordering_inverse(1:super_nonods):', &
                    super_sfc_node_ordering_inverse(1:super_nonods)
! 
           ndim_hilbert=2
           call length_full_super_sfc(full_super_nonods, super_nonods, ndim_hilbert)
           print *,'full_super_nonods=',full_super_nonods
           allocate(full_super_sfc_node_ordering_inverse(full_super_nonods))
! 
           call full_combine_sfcs(full_super_sfc_node_ordering_inverse, &
                 super_sfc_node_ordering_inverse, super_nonods, full_super_nonods) 
            print *,'full_super_nonods, super_nonods, ndim_hilbert:', &
                     full_super_nonods, super_nonods, ndim_hilbert 
            print *,'full_super_sfc_node_ordering_inverse:', &
                     full_super_sfc_node_ordering_inverse
            stop 29219
        endif
! 
! form 1d matrixes and vectors based on no_bc ordering on each grid level...
        max_nonods_sfc_all_grids=5*nonods_no_bc
        if(nfilt_size_sfc==0) then ! use sparse structure to stor matrices on each multi-grid level.
           max_ncola_sfc_all_un=5*ncola_no_bc
           allocate(a_sfc_all_un(max_ncola_sfc_all_un,ncurve))
           allocate(fina_sfc_all_un(max_nonods_sfc_all_grids+1,ncurve))
           allocate(cola_sfc_all_un(max_ncola_sfc_all_un,ncurve))
           allocate(ncola_sfc_all_un(ncurve))
           print *,'max_ncola_sfc_all_un,ncola_no_bc:',max_ncola_sfc_all_un,ncola_no_bc
        else
           allocate(a_sfc(nfilt_size_sfc,max_nonods_sfc_all_grids,ncurve)) 
        endif
        allocate(b_sfc(max_nonods_sfc_all_grids,ncurve))
        allocate(ml_sfc(max_nonods_sfc_all_grids,ncurve))
        max_nlevel=100
        allocate(fin_sfc_nonods(max_nlevel+1))
        do icurve=1,ncurve
           if(nfilt_size_sfc==0) then
!           if(.false.) then
              call best_sfc_mapping_to_sfc_matrix_unstructured( &
                     a_sfc_all_un(:,icurve), fina_sfc_all_un(:,icurve), &
                     cola_sfc_all_un(:,icurve), ncola_sfc_all_un(icurve), &
                     b_sfc(:,icurve), ml_sfc(:,icurve), &
                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                     a_no_bc, b_no_bc, ml_no_bc,  &
                     fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), ncola_no_bc, &
                     nonods_no_bc, max_nonods_sfc_all_grids, &
                     max_ncola_sfc_all_un, max_nlevel) 
               print *,'nlevel:',nlevel
               print *,'nonods_no_bc, nonods_sfc_all_grids:',nonods_no_bc, nonods_sfc_all_grids
               print *,'nonods_sfc_all_grids/nonods_no_bc:',real(nonods_sfc_all_grids)/real(nonods_no_bc)
               print *,'ncola_no_bc, ncola_sfc_all_un(icurve):',ncola_no_bc, ncola_sfc_all_un(icurve)
               print *,'ncola_sfc_all_un(icurve)/ncola_no_bc:',real(ncola_sfc_all_un(icurve))/real(ncola_no_bc)
               print *,'ncola_no_bc/nonods_no_bc:',real(ncola_no_bc)/real(nonods_no_bc)
               print *,'ncola_sfc_all_un(icurve)/nonods_sfc_all_grids:', &
                        real(ncola_sfc_all_un(icurve))/real(nonods_sfc_all_grids)
               rr=0.0
               do ilevel=nlevel,1,-1
                  rr=rr + real( fin_sfc_nonods(nlevel+1) - fin_sfc_nonods(ilevel) )
               end do
               print *,'work for simplified f-net:',real(nonods_sfc_all_grids)
               print *,'work for BunnyNet:',rr
               print *,'BunnyNet/simplied:',rr/real(nonods_sfc_all_grids)
!               stop 2919
           else if(nfilt_size_sfc==3) then
!           if(.false.) then
              call best_sfc_mapping_to_sfc_matrix_3(a_sfc(:,:,icurve), b_sfc(:,icurve), ml_sfc(:,icurve), &
                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                     a_no_bc, b_no_bc, ml_no_bc, relax_keep_off, &
                     fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), ncola_no_bc, &
                     nonods_no_bc, max_nonods_sfc_all_grids, max_nlevel, &
                     nfilt_size_sfc) 
           else
              call best_sfc_mapping_to_sfc_matrix_n(a_sfc(:,:,icurve), b_sfc(:,icurve), ml_sfc(:,icurve), &
                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                     a_no_bc, b_no_bc, ml_no_bc, relax_keep_off, &
                     fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), ncola_no_bc, &
                     nonods_no_bc, max_nonods_sfc_all_grids, max_nlevel, &
                     nfilt_size_sfc) 
           endif
        end do
        print *,'max_nonods_sfc_all_grids,nonods_sfc_all_grids:', &
                 max_nonods_sfc_all_grids,nonods_sfc_all_grids
! 
      iscale_matrices=0 ! default=0 (no scaling) - scale matrices by lumped mass matrix
      i_jacobi_on_finest_sfc=0 ! default=0 - Jacobi on finest sfc matrix. 
      i_jacobi_full_matrix=1 ! defaul=1 - Jacobi on full matirx
! 
!      iscale_matrices=0 ! default=0 (no scaling) - scale matrices by lumped mass matrix
!      i_jacobi_on_finest_sfc=0 ! default=0 - Jacobi on finest sfc matrix. 
!      i_jacobi_full_matrix=1 ! defaul=1 - Jacobi on full matirx
      if(iscale_matrices==1) then
! Scale all the eqns suitable for solver...
        do inod_no_bc=1,nonods_no_bc
           b_no_bc(inod_no_bc) = b_no_bc(inod_no_bc) / ml_no_bc(inod_no_bc)
           do count = fina_no_bc(inod_no_bc), fina_no_bc(inod_no_bc+1)-1 
              a_no_bc(count) = a_no_bc(count) / ml_no_bc(inod_no_bc)
           end do
        end do
! ...scale different levels of the SFC grids...
        do icurve=1,ncurve
           do inod_sfc_all=1,nonods_sfc_all_grids
              b_sfc(inod_sfc_all,icurve) = b_sfc(inod_sfc_all,icurve)   / ml_sfc(inod_sfc_all,icurve)
              if(nfilt_size_sfc.ne.0) then
                 a_sfc(:,inod_sfc_all,icurve)=a_sfc(:,inod_sfc_all,icurve) / ml_sfc(inod_sfc_all,icurve)
              else
                 do count = fina_sfc_all_un(inod_sfc_all,icurve), fina_sfc_all_un(inod_sfc_all+1,icurve)-1
                    a_sfc_all_un(count,icurve) = a_sfc_all_un(count,icurve) / ml_sfc(inod_sfc_all,icurve)
                 end do
              endif
!              print *,'icurve,inod_sfc_all,a_sfc(:,inod_sfc_all,icurve):', &
!                       icurve,inod_sfc_all,a_sfc(:,inod_sfc_all,icurve)
!              print *,'sum(a_sfc(:,inod_sfc_all,icurve)):', &
!                       sum(a_sfc(:,inod_sfc_all,icurve))
           end do
        end do
      endif
!      print *,'minval(a_sfc(2,1:nonods_sfc_all_grids,:)):',minval(a_sfc(2,1:nonods_sfc_all_grids,:))
      print *,'minval(ml_sfc(1:nonods_sfc_all_grids,:)):',minval(ml_sfc(1:nonods_sfc_all_grids,:))
      print *,'minval(ml_sfc(1:nonods_sfc_all_grids,1)):',minval(ml_sfc(1:nonods_sfc_all_grids,1))
!      print *,'minval(ml_sfc(1:nonods_sfc_all_grids,2)):',minval(ml_sfc(1:nonods_sfc_all_grids,2))
!      print *,'maxval(a_sfc(2,1:nonods_sfc_all_grids,:)):',maxval(a_sfc(2,1:nonods_sfc_all_grids,:))
      print *,'maxval(ml_sfc(1:nonods_sfc_all_grids,:)):',maxval(ml_sfc(1:nonods_sfc_all_grids,:))
! 
!      print *,'minval(a_sfc(2,1:nonods_no_bc,:)):',minval(a_sfc(2,1:nonods_no_bc,:))
      print *,'minval(ml_sfc(1:nonods_no_bc,:)):',minval(ml_sfc(1:nonods_no_bc,:))
!      print *,'maxval(a_sfc(2,1:nonods_no_bc,:)):',maxval(a_sfc(2,1:nonods_no_bc,:))
      print *,'maxval(ml_sfc(1:nonods_no_bc,:)):',maxval(ml_sfc(1:nonods_no_bc,:))
!
      print *,'minval(ml),maxval(ml):',minval(ml),maxval(ml)
!      stop 828

! makse diagonal bigger to help convergence 
! e.g. add a bit of mass to help convergence or just make diagonal bigger
!      a_sfc(2,1:nonods_sfc_all_grids,:)=a_sfc(2,1:nonods_sfc_all_grids,:) + 0.5*ml_sfc(1:nonods_sfc_all_grids,:) ! 0.5 works for 200 elements across
!      a_sfc(2,1:nonods_sfc_all_grids,:)=a_sfc(2,1:nonods_sfc_all_grids,:) + 0.3*ml_sfc(1:nonods_sfc_all_grids,:) ! 0.3 works for 200 elements across
!      a_sfc(2,1:nonods_sfc_all_grids,:)=a_sfc(2,1:nonods_sfc_all_grids,:) + 0.25*ml_sfc(1:nonods_sfc_all_grids,:) ! 0.25 does not work for 200 elements across
!      a_sfc(2,:,:)=a_sfc(2,:,:) + 0.2*ml_sfc(:,:) ! 0.2 does not work for 200 elements across
!      a_sfc(2,:,:)=a_sfc(2,:,:)*1.1 ! 1.1 does not work for 200 elements across
!      a_sfc(2,:,:)=a_sfc(2,:,:)*2.0 ! 1.1 works for 200 elements across

! solve the system a_no_bc*psi_no_bc = b_no_bc
        allocate(psi_no_bc(nonods_no_bc), psi_no_bc_guess(nonods_no_bc))
!        nits=2
        nits=2500
        allocate(relax(nlevel))
        allocate(relax_down(nlevel))
!        relax(:)=0.6 ! works slightly rough
!        relax(:)=0.7 ! too rough
!        relax(:)=0.5
!        relax(:)=1.35 !optimal
!        relax(1)=1.35
        relax(:)=1.35
!        relax(1)=1.4
!        relax(:)=1.
!        relax(1)=1.
        relax_down(:)=1.0
        optimal_relax=.true.
        psi_no_bc=0.0
        vcycle = 0
!        fcycle = 1 ! nice new method
        fcycle = 0
        allocate(max_psi_no_bc(nits))
        do its=1,nits
! 
          if(optimal_relax) then ! optimised relaxation method
           if(2*(its/2) == its) then
!           if((2*(its/2) == its).or.(3*(its/3) == its)) then
!              relax(:)=1.7 
              relax(:)=1.9 ! the best with 1.
              relax(1)=2.1 ! the best with 1.
              relax(:)=relax(:)*0.99
!              relax(:)=relax(:)*0.97 ! works for 129x129 not work for 513x513
!              relax(:)=relax(:)*0.95 ! not work for 513x513
!              relax(:)=relax(:)*0.93 ! oscillatory for 513x513
              if(iuse_starting_node==-1) relax(:)=relax(:)*0.9 ! reduce if we are using rapid SFC generation.
!              relax(:)=relax(:)*0.8
           else
              relax(:)=1.0
           endif
           if(96*(its/96) == -its) then
!           if(48*(its/48) == its) then
              relax(:)=1.0
           endif
          endif
! 
               
!           do icurve=2,2
           do icurve=1,ncurve
              psi_no_bc_guess=psi_no_bc
              if(nfilt_size_sfc==0) then ! sparse matrix soln (no approx of multi-grid matrices)
!              if(.false.) then
                if(vcycle==1) then
                 relax(:)=1.0
                 relax_down(:)=relax(:) 
                 call sfc_solver_it_unstructured_vcycle(psi_no_bc, psi_no_bc_guess, a_sfc_all_un(:,icurve), &
                                 fina_sfc_all_un(:,icurve), cola_sfc_all_un(:,icurve), &
                                 ncola_sfc_all_un(icurve), fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, relax_down, &
                                 fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix)  
                else if(fcycle==1) then
!                 do isweep=2,2 ! orginal method
                 do isweep=1,2
                 istart=nlevel-1
                 ifinish=1
                 istep=-1
                 if(isweep==1) then
                    istart=2
                    ifinish=nlevel-1
                    istep=1
                 endif
                 do top_level=nlevel-1,1,-1
!           if(2*((top_level+its)/2) == top_level+its) then
           if(2*((top_level)/2) == top_level) then
!           if((2*(its/2) == its).or.(3*(its/3) == its)) then
!              relax(:)=1.7 
              relax(:)=1.9 ! the best with 1.
              relax(1)=2.1 ! the best with 1.
              relax(:)=relax(:)*0.99
!              relax(:)=relax(:)*0.95
           else
              relax(:)=1.0
           endif
           if(4*(its/4) == -its) then
!           if(48*(its/48) == its) then
              relax(:)=1.0
           endif
!                 top_level=1
!                 relax(:)=1.0
!                 relax_down(:)=relax(:) 
                 psi_no_bc_guess=psi_no_bc
                 call sfc_solver_it_unstructured_fcycle(psi_no_bc, psi_no_bc_guess, a_sfc_all_un(:,icurve), &
                                 fina_sfc_all_un(:,icurve), cola_sfc_all_un(:,icurve), &
                                 ncola_sfc_all_un(icurve), fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 top_level)
                 end do
                 end do
                else 
                 call sfc_solver_it_unstructured(psi_no_bc, psi_no_bc_guess, a_sfc_all_un(:,icurve), &
                                 fina_sfc_all_un(:,icurve), cola_sfc_all_un(:,icurve), &
                                 ncola_sfc_all_un(icurve), fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix)
                endif  
              else if(nfilt_size_sfc==3) then
!              if(.false.) then
                 call sfc_solver_it_3(psi_no_bc, psi_no_bc_guess, a_sfc(:,:,icurve), fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 nfilt_size_sfc)  
              else
                 call sfc_solver_it_n(psi_no_bc, psi_no_bc_guess, a_sfc(:,:,icurve), fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, ncurve_space_fill_curve_numbering(:,icurve), &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 nfilt_size_sfc)  
              endif
           end do
           max_psi_no_bc(its)=maxval(psi_no_bc)
           print *,'its,relax(1),max_psi_no_bc(its):',its,relax(1),max_psi_no_bc(its)
        end do ! do its=1,nits
! 
        open(4, file='sfc-its-error.csv')
!        do its=1,nits
        do its=1,min(300,nits)
!        do its=1,min(2000,nits)
           write(4,*) real(its), abs( max_psi_no_bc(nits) - max_psi_no_bc(its) )
        end do
! map to original ordering psi_no_bc
        allocate(psi(nonods))
        psi=0.0
        do inod=1,nonods
          inod_new = map2_no_bc_nodes(inod)
           if(inod_new.ne.0) psi(inod) = psi_no_bc(inod_new) ! not a b.c. node. 
        end do
! print the soln...
      if(.false.) then
        do j=1,ny
           print *,'j,psi:',j,(psi(inod),inod=(j-1)*nx+1,(j-1)*nx+nx)
        end do
      endif
        open(3, file='sfc-2d-1d.csv')
        j=ny/2 
        do i=1,nx
           inod=(j-1)*nx+i 
           write(3,*) real(i-1)*dx, psi(inod)
        end do
        close(3)
! 
        stop 1928
        end subroutine main7
! 
! 
! 
! The python interface is:
! full_super_nonods = length_full_super_sfc(super_nonods, ndim_hilbert)
        subroutine length_full_super_sfc(full_super_nonods, super_nonods, ndim_hilbert)
        implicit none
        integer, intent( in ) :: super_nonods, ndim_hilbert
        integer, intent( out ) :: full_super_nonods
! Local variables...
        integer ii,nx
! 
        if(ndim_hilbert==1) then ! 1d curve trivial outcomes...
           full_super_nonods=super_nonods
        else ! 2d or 3D Hilbert curve
           do ii=0,100
              nx=2**ii
              full_super_nonods=nx**ndim_hilbert
              if(full_super_nonods.ge.super_nonods) exit
           end do
        endif
        return 
        end subroutine length_full_super_sfc
! 
! 
! 
! The python interface is:
! full_super_sfc_node_ordering_inverse = full_combine_sfcs( &
!                 super_sfc_node_ordering_inverse, super_nonods, full_super_nonods)
        subroutine full_combine_sfcs(full_super_sfc_node_ordering_inverse, &
                 super_sfc_node_ordering_inverse, super_nonods, full_super_nonods) 
! calculate the full_super_sfc_node_ordering_inverse from super_sfc_node_ordering_inverse 
! and the lengths of these variables super_nonods, full_super_nonods
! It does this by tracing the super_sfc_node_ordering_inverse forwards then backwards 
! filling in full_super_sfc_node_ordering_inverse until full_super_nonods are used up. 
        implicit none
        integer, intent( in ) :: super_nonods, full_super_nonods
        integer, intent( in ) :: super_sfc_node_ordering_inverse(super_nonods)
        integer, intent( out ) :: full_super_sfc_node_ordering_inverse(full_super_nonods)
! local variables...
        integer iforward_backward, istart, ifinish, istep, ifull_super_nod, isuper_nod
        logical done
        ifull_super_nod=0
        do iforward_backward=1,1000
           if( 2*(iforward_backward/2) == iforward_backward) then
              istart=super_nonods-1
              ifinish=1
              istep=-1
           else
              istart=1
              ifinish=super_nonods
              istep=1
           endif
!              print *,'istart=',istart
           done=.false.
           do isuper_nod=istart, ifinish, istep
              ifull_super_nod=ifull_super_nod+1
              full_super_sfc_node_ordering_inverse(ifull_super_nod) &
                 =super_sfc_node_ordering_inverse(isuper_nod)
              if(ifull_super_nod==full_super_nonods) then
                done=.true.
                exit
             endif
           end do
           if(done) exit
        end do
!            print *,'full_super_nonods, super_nonods, ndim_hilbert:', &
!                     full_super_nonods, super_nonods, ndim_hilbert 
!            print *,'full_super_sfc_node_ordering_inverse:', &
!                     full_super_sfc_node_ordering_inverse
        return 
        end subroutine full_combine_sfcs
! 
! 
! The python interface is: 
! super_sfc_node_ordering_inverse, super_nonods = combine_sfcs( &
!                         sfc_node_ordering,nonods,ncurve) 
        subroutine combine_sfcs(super_sfc_node_ordering_inverse, super_nonods, &
                         sfc_node_ordering,nonods,ncurve)
! this subroutine combines SFCs and with ncurve of them. 
! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
        implicit none
        integer, intent( in ) :: nonods,ncurve
        integer, intent( in ) :: sfc_node_ordering(nonods,ncurve)
        integer, intent( out ) :: super_sfc_node_ordering_inverse(2*ncurve*nonods)
        integer, intent( out ) :: super_nonods
! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:,:)
        integer, allocatable :: connect_start_sfc(:,:),connect_finish_sfc(:,:)
        integer icurve,jcurve, inod,inod_sfc
        integer inod_icurve,inod_jcurve, inod_jcurve_sfc
        integer isuper_nod_sfc
! 
        allocate(sfc_node_ordering_inverse(nonods,ncurve))
        allocate(connect_start_sfc(ncurve,ncurve),connect_finish_sfc(ncurve,ncurve))
        do icurve=1,ncurve
           do inod=1,nonods
              inod_sfc = sfc_node_ordering(inod,icurve) 
              sfc_node_ordering_inverse(inod_sfc,icurve) = inod
           end do 
        end do 
! 
! Limited brute force search for the best ordering in terms of path length minimum.
! Just switch around from the start and finish. 
        icurve=1
        super_sfc_node_ordering_inverse(1:nonods)=sfc_node_ordering_inverse(1:nonods,icurve)
        isuper_nod_sfc=nonods
        jcurve=icurve+1
        do inod_sfc=nonods-1,1,-1 ! trace back to node 1 of next SFC... 
           inod_icurve = sfc_node_ordering_inverse(inod_sfc,icurve)
           inod_jcurve_sfc = sfc_node_ordering(inod_icurve,jcurve)
           isuper_nod_sfc = isuper_nod_sfc + 1
           super_sfc_node_ordering_inverse(isuper_nod_sfc)=inod_icurve
           if(inod_jcurve_sfc==1) exit
        end do
! 
! for second SFC...
        icurve=2
        do inod_sfc=2,nonods ! trace forward from SFC node 1... 
           inod_icurve = sfc_node_ordering_inverse(inod_sfc,icurve)
           isuper_nod_sfc = isuper_nod_sfc + 1
           super_sfc_node_ordering_inverse(isuper_nod_sfc)=inod_icurve
        end do
! now trace back to node 1 of next SFC.
        if(ncurve==3) then
           jcurve=icurve+1 
           do inod_sfc=nonods-1,1,-1 ! trace back to node 1 of next SFC... 
              inod_icurve = sfc_node_ordering_inverse(inod_sfc,icurve)
              inod_jcurve_sfc = sfc_node_ordering(inod_icurve,jcurve)
              isuper_nod_sfc = isuper_nod_sfc + 1
              super_sfc_node_ordering_inverse(isuper_nod_sfc)=inod_icurve
              if(inod_jcurve_sfc==1) exit
           end do
! for second SFC...
           icurve=3
           do inod_sfc=2,nonods ! trace back to node 1 of next SFC... 
              inod_icurve = sfc_node_ordering_inverse(inod_sfc,icurve)
              isuper_nod_sfc = isuper_nod_sfc + 1
              super_sfc_node_ordering_inverse(isuper_nod_sfc)=inod_icurve
           end do
        endif ! if(ncurve==3) then
        super_nonods=isuper_nod_sfc
! 
        return
        end subroutine combine_sfcs
! 
! 
! 
! 
! The python interface is called from python with:
!     xc,yc, mesh_functional, voronoi_cell_connect, number_voronoi_cells = voronoi_cell_functional_and_pts(
!                                             color, dx, dy, desired_edge_length, &
!                                             nx,ny, max_voronoi_cells, max_connect ) 
          subroutine voronoi_cell_functional_and_pts(xc,yc, mesh_functional, &
                                             voronoi_cell_connect, number_voronoi_cells, &
                                             color, dx, dy, desired_edge_length, &
                                             nx,ny, max_voronoi_cells, max_connect ) 
          implicit none
! mesh_functional: This subdoutine calculates the quality if the triangle mesh mesh_functional from 
! a grid gof pixes of different color defined by color(i,j)=colour of pixel (integer) and 
! the grid or image has dimensions (nx,ny) - number of cells across and down respectively. 
! xc,yc: It calculates the centres of mass of the Voronoi cells (xc(icell),yc(icell)) in which icell 
! is the cell number. 
! number_voronoi_cells: It calculates the number of Voronoi cells number_voronoi_cells. 
! dx, dy: are the dimensions of the cells. 
! desired_edge_length is the desired edge length of the triangles. 
! max_voronoi_cells contains the maximum number of Voronoi cells in the image. 
! max_connect is the maximum number of Voronoi cells surrounding (next too) a given Voronoi cell. 
! voronoi_cell_connect(icell, jj)= the jj^th index to a aurrounding Voronoi cell of cell icell. 
! voronoi_cell_connect(icell, jj) contains a non-zero only up to the number jj of surrounding 
! Voronoi cells otherwise it contains 0. 
          integer, intent(in) :: nx,ny, max_voronoi_cells, max_connect
          real, intent(out) :: xc(max_voronoi_cells), yc(max_voronoi_cells)
          real, intent(out) :: mesh_functional
          integer, intent(out) :: voronoi_cell_connect(max_voronoi_cells,max_connect)
          integer, intent(out) :: number_voronoi_cells
          real, intent(in) :: dx, dy, desired_edge_length
          integer, intent(in) :: color(nx,ny)
! Local variables
          integer, allocatable :: mark_centre_extend(:,:), mark_voronoi_no_all_extend(:,:)
          integer, allocatable :: color_extend(:,:), color2_extend(:,:), boundary_extend(:,:)
          integer, allocatable :: xy_count(:), voronoi_cell_color(:) 
          integer i,j, ii,jj, idiff2_color, iswitch, idiff_bound, iconnect, imax, ivor
          integer idiff_color, idiff_mark, imax_mark, jvoronoi, ivoronoi, its,nits
! 
          allocate(color_extend(0:nx+1,0:ny+1), color2_extend(0:nx+1,0:ny+1))
          allocate(mark_centre_extend(0:nx+1,0:ny+1))
          allocate(mark_voronoi_no_all_extend(0:nx+1,0:ny+1))
          allocate(boundary_extend(0:nx+1,0:ny+1))
          allocate(voronoi_cell_color(max_voronoi_cells))
! extend the values to produce borders...
          call extend_2_borders(color_extend,color,nx,ny) 
! 
! indicator of the boundaries
          boundary_extend(:,:)=1 
          boundary_extend(1:nx,1:ny)=0 
! 
          nits=2*max(nx,ny)
! 
! Shrink the Voronoi cell colours to a single point. 
          color2_extend(:,:)=0
          mark_centre_extend(:,:)=0
          number_voronoi_cells=0
          do its=1,nits
             do i=1,nx
             do j=1,ny 
                if(color2_extend(i,j)==0) then ! define color2_extend(i,j)
                   idiff_color =0 
                   idiff_bound =0
                   idiff2_color=0 
                   iswitch=0
                   do jj=-1,1
                   do ii=-1,1
                      idiff_color  = idiff_color &
                        + (1-boundary_extend(i+ii,j+jj))*min(1, abs( color_extend(i,j)-color_extend(i+ii,j+jj) )  )
                      idiff_bound = idiff_bound + boundary_extend(i+ii,j+jj)
                      if( (1-boundary_extend(i+ii,j+jj))*(its-color2_extend(i+ii,j+jj)) == 1 ) then
                         if( ii*jj .ne. 0 ) then
                         if( color_extend(i,j)-color_extend(i+ii,j+jj) == 0 ) then
                            iswitch=1
                         endif
                         endif
                      endif
                      idiff2_color = idiff2_color + (1-boundary_extend(i+ii,j+jj))*min(1, color2_extend(i+ii,j+jj) )
                   end do
                   end do
                   if(its==1) then
                      if(idiff_color.ge.1) color2_extend(i,j)=1
                   else 
                      if(iswitch==1) color2_extend(i,j)=its
                   endif
                   if(color2_extend(i,j).ne.0) then
! is color issolated then if so its a centre of a Voronoi cell. 
                      if(idiff2_color+idiff_bound==8) then ! cell surrounded by claimed cells
                         number_voronoi_cells=number_voronoi_cells+1
                         mark_centre_extend(i,j)=number_voronoi_cells
                         voronoi_cell_color(number_voronoi_cells)=color_extend(i,j)
                      endif
                   endif ! if(color2_extend(i,j).ne.0) then
                endif ! if(color2_extend(i,j).ne.0) then
             end do
             end do
             call extend_2_borders_only( color2_extend, color2_extend, nx,ny) 
          end do
       if(.true.) then
          print *,'number_voronoi_cells:',number_voronoi_cells
! 
          print *,'color_extend:'
          do j=0,ny+1
             print *,'j,color_extend(:,j):',j,':',color_extend(:,j)
          end do
!
          print *,'color2_extend:'
          do j=0,ny+1
             print *,'j,color2_extend(:,j):',j,':',color2_extend(:,j)
          end do
! 
          print *,'mark_centre_extend:'
          do j=1,ny
             print *,'j,mark_centre_extend(:,j):',j,':',mark_centre_extend(:,j)
          end do
       endif
!          stop 292
! 
! Grow the cells from their centre.  mark_voronoi_no_all(i,j) = voronoi cell number for cell (i,j). 
          mark_voronoi_no_all_extend(:,:)=mark_centre_extend(:,:)
          do its=1,nits
             do j=1,ny 
             do i=1,nx
                if(mark_voronoi_no_all_extend(i,j)==0) then ! define mark_voronoi_no_all_extend(i,j)
                   idiff_color=0 
!                   idiff_mark=0 
                   imax_mark=0 
                   do jj=-1,1
                   do ii=-1,1
                      idiff_color  = idiff_color + min(1, abs( color_extend(i,j)-color_extend(i+ii,j+jj) )  )
!                      idiff_mark = idiff_mark &
!                       + abs( mark_voronoi_no_all_extend(i,j)-mark_voronoi_no_all_extend(i+ii,j+jj) )
                      if(color_extend(i,j)-color_extend(i+ii,j+jj)==0) imax_mark = &
                             max(imax_mark, mark_voronoi_no_all_extend(i+ii,j+jj))
!                      imax_mark = max(imax_mark, mark_voronoi_no_all(i+ii,j+jj))
                   end do
                   end do
! Have at least one color of the same colour next to cell (i,j) i.e. idiff_color.le.7
                   if(idiff_color.le.7) then
                      mark_voronoi_no_all_extend(i,j)=imax_mark
                   endif
                endif ! if(mark_voronoi_no_all_extend(i,j)==0) then
             end do
             end do
             call extend_2_borders_only( mark_voronoi_no_all_extend, mark_voronoi_no_all_extend, nx,ny) 
          end do
! 
       if(.true.) then
          print *,'dx,dy:',dx,dy
! 
          print *,'mark_voronoi_no_all_extend:'
          do j=1,ny
             print *,'j,mark_voronoi_no_all_extend(:,j):',j,':',mark_voronoi_no_all_extend(:,j)
          end do
!          stop 2821
       endif
! 
! Work out real centre of mass of each voronoi cell.
        print *,'here0'
          allocate(xy_count(number_voronoi_cells))
          xc(1:number_voronoi_cells)=0.0
          yc(1:number_voronoi_cells)=0.0
          xy_count=0
          do j=1,ny 
          do i=1,nx
!             print *,'i,j:',i,j
             ivoronoi = mark_voronoi_no_all_extend(i,j) 
             xc(ivoronoi)=xc(ivoronoi)+ dx*real(i-1) + 0.5*dx
             yc(ivoronoi)=yc(ivoronoi)+ dy*real(j-1) + 0.5*dy
             xy_count(ivoronoi)=xy_count(ivoronoi)+1
          end do
          end do
          xc(1:number_voronoi_cells)=xc(1:number_voronoi_cells)/real( xy_count(1:number_voronoi_cells) )
          yc(1:number_voronoi_cells)=yc(1:number_voronoi_cells)/real( xy_count(1:number_voronoi_cells) )
          print *,'here1'
! 
! Make a connectivity list...
          voronoi_cell_connect=0 
          do j=1,ny 
          do i=1,nx
             do jj=-1,1
             do ii=-1,1
                if(abs( mark_voronoi_no_all_extend(i,j)-mark_voronoi_no_all_extend(i+ii,j+jj) )>0) then ! add to connectivity list
                   call put_in_connect_for_voronoi_cell( &
                             voronoi_cell_connect(mark_voronoi_no_all_extend(i,j),:), &
                             mark_voronoi_no_all_extend(i+ii,j+jj), max_connect) 
                endif
             end do
             end do
          end do
          end do
          print *,'here2'
! 
! Calculate the functional mesh_functional which says what the quality of the mesh is. 
          mesh_functional=0.0
          do ivoronoi=1,number_voronoi_cells
             do jj=1,max_connect
                jvoronoi=voronoi_cell_connect(ivoronoi,jj)
                if(jvoronoi==0) exit
                if(jvoronoi.ne.ivoronoi) then
                   mesh_functional = mesh_functional &
                   + (sqrt((xc(jvoronoi)-xc(ivoronoi))**2 + (yc(jvoronoi)-yc(ivoronoi))**2) &
                        - desired_edge_length)**2
                endif
             end do
          end do
          mesh_functional = mesh_functional / real(number_voronoi_cells)
          print *,'here3'
          do ivor=1,number_voronoi_cells
             print *,'ivor,xc(ivor),yc(ivor):',ivor,xc(ivor),yc(ivor)
             imax=0
             do iconnect=max_connect,1,-1
                if(voronoi_cell_connect(ivor,iconnect)==0) imax=iconnect
             end do
             print *,'voronoi_cell_connect(ivor,:):', &
                 (voronoi_cell_connect(ivor,iconnect),iconnect=1,imax-1)
          end do
          print *,'here4'
          return
          end subroutine voronoi_cell_functional_and_pts
! 
! 
! 
! 
          subroutine extend_2_borders(color_extend,color,nx,ny) 
! extend the values to produce borders...
! if only_boundaries==0 then update the interior else only update the boundaries
          integer, intent(in) :: nx,ny
          integer, intent(in) :: color(nx,ny)
          integer, intent(out) :: color_extend(0:nx+1,0:ny+1)
! 
          color_extend(1:nx,1:ny)=color(1:nx,1:ny)
! 4 boundaries...
          color_extend(0,1:ny)=color(1,1:ny)
          color_extend(nx+1,1:ny)=color(nx,1:ny)
          color_extend(1:nx,0)=color(1:nx,1)
          color_extend(1:nx,ny+1)=color(1:nx,ny)
! 4 corners...
          color_extend(0,0)=color(1,1)
          color_extend(nx+1,ny+1)=color(nx,ny)
          color_extend(nx+1,0)=color(nx,1)
          color_extend(0,ny+1)=color(1,ny)
          return
          end subroutine extend_2_borders
! 
! 
! 
          subroutine extend_2_borders_only(color_extend,color,nx,ny) 
! extend the values to produce borders only...
! if only_boundaries==0 then update the interior else only update the boundaries
          integer, intent(in) :: nx,ny
          integer, intent(in) :: color(0:nx+1,0:ny+1)
          integer, intent(out) :: color_extend(0:nx+1,0:ny+1)
! 
! 4 boundaries...
          color_extend(0,1:ny)=color(1,1:ny)
          color_extend(nx+1,1:ny)=color(nx,1:ny)
          color_extend(1:nx,0)=color(1:nx,1)
          color_extend(1:nx,ny+1)=color(1:nx,ny)
! 4 corners...
          color_extend(0,0)=color(1,1)
          color_extend(nx+1,ny+1)=color(nx,ny)
          color_extend(nx+1,0)=color(nx,1)
          color_extend(0,ny+1)=color(1,ny)
          return
          end subroutine extend_2_borders_only
! 
! 
! 
! 
         subroutine put_in_connect_for_voronoi_cell(voronoi_cell_connect, &
                      mark_voronoi_no_nab, max_connect) 
! Insert mark_voronoi_no_nab into the list voronoi_cell_connect if its not already in it. 
         implicit none
         integer, intent(in) :: max_connect
         integer, intent(in) :: mark_voronoi_no_nab
         integer, intent(out) :: voronoi_cell_connect(max_connect)
! local variables...
!         logical found
         integer i,ikeep
! 
!         found = .false.
         ikeep=0
         do i=1,max_connect
            if(voronoi_cell_connect(i)==0) then
               ikeep=i
               exit
            endif
            if(voronoi_cell_connect(i)==mark_voronoi_no_nab) then ! found this cell already in list
!               found=.true.
               exit
            endif
         end do
         if(ikeep.ne.0) then
            voronoi_cell_connect(ikeep)=mark_voronoi_no_nab
         endif

         return
         end subroutine put_in_connect_for_voronoi_cell
! 
! 
!           
! 
          subroutine ncurve_python_subdomain_space_filling_curve_stub( ncurve_whichd,  &
           ncurve_space_fill_curve_numbering,  cola,fina, iuse_starting_node, &
           graph_trim, ncurve, nonods,ncola)
! *******************************************************************************************************
! This subroutine uses nested disection of the domain into subdomains in order to form 
! a space filling curve: space_fill_curve_numbering. whichd contains the subdomain ordering.
! It also form ncurve space filling curves and puts them in ncurve_space_fill_curve_numbering(nod,icurve)
! in which nod is the node number (original) and icurve is the space filling curve number.
! If iuse_starting_node>0 then use the starting node from the end of the previous space filling curve. 
! abs(graph_trim) is the graph trimming option for >1 space filling curve numbers. 
! abs(graph_trim)>3 and <9 are use all the graph trimming (original method).
! if graph_trim<0 use a matrix a and set to large values the graph weights we want to discoursage going through.
!          no_trim=.false.; trimmed_graph_for_decomposition_only=.false.; trimmed_graph_for_reorder_only=.false.
!          duplicate=.false.; got_a_matrix=.false.
!          if(abs(graph_trim)==0).or.(abs(graph_trim)==10)) no_trim=.true.
!          if((abs(graph_trim)==1).or.(abs(graph_trim)==2)) then
!             trimmed_graph_for_decomposition_only=.true.
!             if(abs(graph_trim)==1) duplicate=.true.
!          endif
!          if(abs(graph_trim)==3) trimmed_graph_for_reorder_only=.true.
! abs(trim_graph)=0 or abs(graph_trip)10 dont trim graph. 
! graph_trim=1 then extended cola to duplicate connections and mimick greater weights for domain decomp only.
! graph_trim=2 trim graph only for domain decomposition
! graph_trim=3 trim graph only for re-ordering optimization (no as good) 
! graph_trim>=4 trim graph for decomposition and reordering. 
! *******************************************************************************************************
          implicit none
          INTEGER, INTENT(IN) :: iuse_starting_node, graph_trim, ncurve, nonods,ncola
          INTEGER, INTENT(out) :: ncurve_whichd(nonods,ncurve)
          INTEGER, INTENT(out) :: ncurve_space_fill_curve_numbering(nonods,ncurve)
          INTEGER, INTENT(in) :: cola(ncola),fina(nonods+1)
          print *,'should not be running it like this stopping'
          print *,'call the subroutine ncurve_python_subdomain_space_filling_curve '
          print *,'and not ncurve_python_subdomain_space_filling_curve_stub'
          stop 3921
          return
          end subroutine ncurve_python_subdomain_space_filling_curve_stub
! 
! 
! 
! The python interface is:
!    add_ncurve_space_fill_curve_numbering, n_add_ncurve = add_sfc_for_matrix(
!           ncurve_space_fill_curve_numbering,  cola,fina,  &
!           ncurve, nonods,ncola, max_add_ncurve)
          subroutine add_sfc_for_matrix(add_ncurve_space_fill_curve_numbering, &
           n_add_ncurve, &
           ncurve_space_fill_curve_numbering,  cola,fina,  &
           ncurve, nonods,ncola, max_add_ncurve)
! *******************************************************************************************************
! This subroutine uses the SFCs to form the matricies and adds more SFCs to make up for 
! lack of variables within the matrix a. 
! It also form ncurve space filling curves and puts them in ncurve_space_fill_curve_numbering(nod,icurve)
! in which nod is the node number (original) and icurve is the space filling curve number.
! If iuse_starting_node>0 then use the starting node from the end of the previous space filling curve. 
! abs(graph_trim) is the graph trimming option for >1 space filling curve numbers. 
! abs(graph_trim)>3 and <9 are use all the graph trimming (original method).
! if graph_trim<0 use a matrix a and set to large values the graph weights we want to discoursage going through.
!          no_trim=.false.; trimmed_graph_for_decomposition_only=.false.; trimmed_graph_for_reorder_only=.false.
!          duplicate=.false.; got_a_matrix=.false.
!          if(abs(graph_trim)==0).or.(abs(graph_trim)==10)) no_trim=.true.
!          if((abs(graph_trim)==1).or.(abs(graph_trim)==2)) then
!             trimmed_graph_for_decomposition_only=.true.
!             if(abs(graph_trim)==1) duplicate=.true.
!          endif
!          if(abs(graph_trim)==3) trimmed_graph_for_reorder_only=.true.
! abs(trim_graph)=0 or abs(graph_trip)10 dont trim graph. 
! graph_trim=1 then extended cola to duplicate connections and mimick greater weights for domain decomp only.
! graph_trim=2 trim graph only for domain decomposition
! graph_trim=3 trim graph only for re-ordering optimization (no as good) 
! graph_trim>=4 trim graph for decomposition and reordering. 
! *******************************************************************************************************
          implicit none
          INTEGER, INTENT(in) :: max_add_ncurve
          INTEGER, INTENT(IN) :: ncurve, nonods,ncola
          INTEGER, INTENT(in) :: ncurve_space_fill_curve_numbering(nonods,ncurve)
          INTEGER, INTENT(in) :: cola(ncola),fina(nonods+1)
          INTEGER, INTENT(out) :: add_ncurve_space_fill_curve_numbering(nonods,max_add_ncurve)
          INTEGER, INTENT(out) :: n_add_ncurve
! Local variables
          integer, allocatable :: sfc_node_ordering_inverse(:,:),which_sfc_cola(:)
          integer icurve,inod,inod_sfc,nfilter,jnod,jnod_sfc,count,jnod_look
          integer acc_sfc,count2,inod2
          print *,'inside add_sfc_for_matrix'
! ncurve_space_fill_curve_numbering(nod) = new node numbering from current node number nod.
          allocate(sfc_node_ordering_inverse(nonods,ncurve))
          allocate(which_sfc_cola(ncola))
          do icurve=1,ncurve
             do inod=1,nonods
                inod_sfc = ncurve_space_fill_curve_numbering(inod,icurve) 
                sfc_node_ordering_inverse(inod_sfc,icurve) = inod
             end do 
          end do
! 
          nfilter=3
          which_sfc_cola(:)=0
          do icurve=1,ncurve
             do inod_sfc=1,nonods
                inod = sfc_node_ordering_inverse(inod_sfc,icurve)
                do jnod_sfc=max(1,inod_sfc - nfilter/2), min(nonods,inod_sfc + nfilter/2)
                   jnod = sfc_node_ordering_inverse(jnod_sfc,icurve)
                   do count=fina(inod),fina(inod+1)-1
                      jnod_look = cola(count)
                      if(jnod==jnod_look) then
                         if(which_sfc_cola(count)==0) then
                            which_sfc_cola(count)=icurve
                            do count2=fina(jnod),fina(jnod+1)-1 ! label the transpose...
                               inod2=cola(count2)
                               if(inod==inod2) which_sfc_cola(count2)=icurve
                            end do
                         endif
                      endif
                   end do
                end do ! do jnod_sfc=max(1,inod_sfc - nfilter/2), min(nonods,inod_sfc + nfilter/2)
             end do ! do inod_sfc=1,nonods
          end do ! do icurve=1,ncurve
! 
! do some counting...
          print *,'nonods,ncola,nfilter=',nonods,ncola,nfilter
          do icurve=0,ncurve
             acc_sfc=0
             do count=1,ncola
                if(which_sfc_cola(count)==icurve) acc_sfc=acc_sfc+1
             end do
             print *,'icurve,acc_sfc:',icurve,acc_sfc
          end do
! 
! Add some more SFC curves...
          print *,'max_add_ncurve:',max_add_ncurve
          add_ncurve_space_fill_curve_numbering(:,:)=0 
          do icurve=1,max_add_ncurve
             print *,'*** icurve=',icurve
             acc_sfc=0
             do inod=1,nonods
                do count=fina(inod),fina(inod+1)-1
                   jnod = cola(count)
                   if(which_sfc_cola(count)==0) then 
! try to make inod and jnod two consecutuve nodes in the sfc. 
                      inod_sfc = add_ncurve_space_fill_curve_numbering(inod,icurve) 
                      jnod_sfc = add_ncurve_space_fill_curve_numbering(jnod,icurve) 
                      if((inod_sfc==0).and.(jnod_sfc==0)) then ! make both on SFC...
                         acc_sfc = acc_sfc + 1
                         add_ncurve_space_fill_curve_numbering(inod,icurve)=acc_sfc
                         acc_sfc = acc_sfc + 1
                         add_ncurve_space_fill_curve_numbering(jnod,icurve)=acc_sfc
                         which_sfc_cola(count)=-icurve
                         do count2=fina(jnod),fina(jnod+1)-1 ! label the transpose...
                            inod2=cola(count2)
                            if(inod==inod2) which_sfc_cola(count2)=-icurve
                         end do
                      endif ! if(jnod_sfc==0) then
                   endif ! if(which_sfc_cola(count)==0) then 
                end do ! do count=fina(inod),fina(inod+1)-1
             end do ! do inod=1,nonods
             print *,'nonods,acc_sfc:',nonods,acc_sfc
             if(acc_sfc==0) then
                n_add_ncurve=icurve-1
                exit
             endif
! fill in the rest of the SFC...
             do inod=1,nonods
                if(add_ncurve_space_fill_curve_numbering(inod,icurve)==0) then
                   acc_sfc = acc_sfc + 1
                   add_ncurve_space_fill_curve_numbering(inod,icurve)=acc_sfc
                endif
             end do
             print *,'should be the same ::: nonods,acc_sfc:',nonods,acc_sfc
          end do ! do icurve=1,max_add_ncurve
          print *,'n_add_ncurve=',n_add_ncurve
          
          stop 3923
          return
          end subroutine add_sfc_for_matrix
!  
! 
! 
! 
        subroutine remove_bc_nodes_a_b(a_no_bc, b_no_bc, ml_no_bc, fina_no_bc, cola_no_bc, ncola_no_bc, &
                   map2_no_bc_nodes, a, b, ml, fina, cola,ion_boundary, nonods_no_bc, ncola, nonods) 
! this subroutine takes away the b.c.s from the matrix eqns and forms 
        integer, intent( in ) :: nonods_no_bc, ncola, nonods
        integer, intent( out ) :: ncola_no_bc
        real, intent( out ) :: a_no_bc(ncola), b_no_bc(nonods_no_bc), ml_no_bc(nonods_no_bc)
        integer, intent( out ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola)
        integer, intent( out ) :: map2_no_bc_nodes(nonods)
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( in ) :: fina(nonods+1), cola(ncola), ion_boundary(nonods)
! local variables...
        integer inod,inod_new, jcol,jcol_new, count,count2
!  
        map2_no_bc_nodes=0
        inod_new=0
        do inod=1,nonods
           if(ion_boundary(inod)==0) then ! not on boundary of domain
              inod_new=inod_new+1
              map2_no_bc_nodes(inod)=inod_new
           endif
        end do
        print *,'inod_new:',inod_new
! Take away boundary nodes...
        count2=0
        do inod=1,nonods
           inod_new=map2_no_bc_nodes(inod)
           if(inod_new.ne.0) then
              b_no_bc(inod_new)=b(inod)
              ml_no_bc(inod_new)=ml(inod)
              fina_no_bc(inod_new)=count2+1
              do count=fina(inod),fina(inod+1)-1
                 jcol=cola(count)
                 jcol_new=map2_no_bc_nodes(jcol)
                 if(jcol_new.ne.0) then
                    count2=count2+1
                    a_no_bc(count2)=a(count)
                    cola_no_bc(count2)=jcol_new
                 endif
              end do
           endif
        end do
        ncola_no_bc=count2
        fina_no_bc(nonods_no_bc+1)=ncola_no_bc+1
        return 
        end subroutine remove_bc_nodes_a_b
! 
! 
! 
! 
! python interface: 
! filt_nxnx, filt_nnx, ml = filters_for_structured_mesh_dg(dx,ndim,nloc)
        subroutine main4
        implicit none
! this subroutine calculates filters for the 2D rectangular element and 
! the 3D hex element of dimensions dx.
! filt_nxnx is the filter for the diffusion/Laplacian operator. 
! filt_nnx is the filter for the derivatives in the x,y and z-directions. 
! ml contains the mass associated with the local nodes. 
! nloc=no of nodes per element
! ndim=no of dimensions 
! dx is the dimensions of the element - width, length, height. 
! For the arrays:
! filt_nxnx(3,3,1+(ndim-2)*2, nloc)
! filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! The 3,3 is the dimensions of the filter in 2D. It needs to be a 3x3 array.
! NDIM is the number of dimensions ndim=2 in our case but will change to3 soon.
! nloc is the number of local noes in an element. So we produce a filter for each local node of an element - 4 in our case.
! nsign=1 or 2 and is 1 is we have a positive sign of the component of interest (idim that goes into
! the ndim part of the array) of the velocity. Its 2 if the component is negative. 
        integer, parameter :: nsign=2,nloc=8,ndim=3 ! 3D
!        integer, parameter :: nsign=2,nloc=4,ndim=2 ! 2D
        real dx(ndim)
        real filt_nxnx(3,3,1+(ndim-2)*2, nloc)
        real filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
        real ml(nloc)
! python interface: 
! filt_nxnx, filt_nnx, ml = filters_for_structured_mesh_dg(dx,ndim,nloc)
! local variables...
        integer i,j,k,idim,isign,iloc
! 
        dx=1.0
        call filters_for_structured_mesh_dg(filt_nxnx, filt_nnx, ml, dx,ndim,nloc)

        print *,'diffusion:'
        do iloc=1,nloc
           do k=1,1+2*(ndim-2)
           do j=3,1,-1
              print *,'j,iloc,filt_nxnx(:,j,k,iloc):',j,iloc,filt_nxnx(:,j,k,iloc)
           end do
           end do
           print *,' '
        end do
! 
!        print *,' '
        print *,'advection:'
        do iloc=1,nloc
        do isign=1,nsign
        do idim =1,ndim
           print *,' '
           do k=1,1+2*(ndim-2)
           do j=3,1,-1
       print *,'j,idim,isign,iloc:',j,idim,isign,iloc 
       print *,'filt_nnx(:,j,k,idim,isign,iloc):',filt_nnx(:,j,k,idim,isign,iloc)
           end do
           end do
        end do
        end do
        end do


!        print *,'filt_nxnx:',filt_nxnx
!        print *,'filt_nnx:',filt_nnx
        print *,'ml:',ml
        stop 2829
        end subroutine main4
! 
! 
! 
!        programsubroutine main ! ann filter and simple code
        subroutine main3 ! ann filter and simple code
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
! nx=no of nodes across, ny=no of nodes up. 
        integer, parameter :: nx=11,ny=11, nonods=nx*ny, totele=(nx-1)*(ny-1)
        integer, parameter :: nloc=4,sngi=3, ngi=9, ndim=2, &
                              nface=4, max_face_list_no=2
        integer, parameter :: ntime = 10
        real, parameter :: dx=1.0, dy=1.0, dt=0.1 
! example of node ordering...
!      7-----8-----9
!      !     !     !
!      !     !     !        
!      4-----5-----6
!      !     !     !
!      !     !     !        
!      1-----2-----3
        real b(nonods), ml(nonods)
        real k(ndim,nonods), sig(nonods), s(nonods), t_new(nonods), t_old(nonods)
        real u(ndim,nonods)
        real x_all(ndim,nloc*totele)
        integer fina(nonods+1) 
        integer ndglno(nloc*totele)
! filters for ann:
        integer, allocatable :: cola(:) 
        real, allocatable :: a(:),a_filter(:) 
        integer npoly,ele_type, iloc,count, ncola
        integer enx_i, eny_j, ele, i,j, ii,jj, iii,jjj, inod, jnod, itime
! 
        ncola=9*nonods ! this is the maximum size of ncola it will be smaller. 
        allocate(a(ncola),a_filter(ncola))
        allocate(cola(ncola))
! 
        print *,'here in main'
        stop 23
! 
        call set_up_grid_problem(t_old, t_new, x_all, ndglno, fina, cola, &
                                 dx, dy, ncola, nx, ny, nloc, ndim)

        npoly=1
        ele_type=1
        k(:,:)=0.0; sig=0.0; s=0.0; u(1,:)=1.0; u(2,:)=1.0
!        do inod=1,nonods
!           if(
!        end do
! in python:
! a, b = u2r.get_fe_matrix_eqn(x_all, u, k, sig, s, fina,cola, ncola, ndglno, nonods,totele,nloc, ndim, ele_type)
! ml = the lumped mass 
        call get_fe_matrix_eqn(a,b, ml, k, sig, s, u, x_all, &
                               fina,cola, ndglno,  &
                               ele_type, ndim, totele,nloc, ncola,nonods) 
! this subroutine finds the matrix eqns A T=b - that is it forms matrix a and vector b and the soln vector is T 
! in python:
! a_filter = get_filter_matrix( a, ml, dt, fina,cola, ncola,nonods)
! This subroutine finds the matrix eqns a_filter = -M_L^{-1} ( -M_L + dt* A). 
! Time stepping can be realised with T^{n+1} = a_filter * T^n 
        call get_filter_matrix(a_filter, a, ml, dt,  &
                               fina,cola, ncola,nonods)

! in python:
! t_new = time_step_filter_matrix( t_old, a_filter, fina,cola, ncola,nonods)
        do itime=1,ntime
! set bcs...
           do j=1,ny
           do i=1,nx
              inod=(j-1)*nx+i
              if((i==1).or.(i==nx)) t_old(inod)=0.0
              if((j==1).or.(j==ny)) t_old(inod)=0.0
           end do
           end do

           call time_step_filter_matrix(t_new, t_old, a_filter,   &
                                        fina,cola, ncola, nonods)
           t_old=t_new
        end do
        do j=ny,1,-1
           print *,'j,t_new( (j-1)*nx+1: (j-1)*nx +nx):',j
           print *,t_new( (j-1)*nx+1: (j-1)*nx +nx)
        end do
        stop 2922
        end subroutine main3
! 
! 
! 
! 
        subroutine cty_filter_orig ! ann filter and simple code
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
 !       integer, parameter :: totele=1,nloc=4,snloc=2,sngi=3, ngi=9, ndim=2, &
 !                             nface=4, max_face_list_no=2
! 3d:
       integer, parameter :: totele=1,nloc=8,snloc=4,sngi=9, ngi=27, ndim=3, &
                              nface=6, max_face_list_no=4
        real :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real :: face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no)
        real :: face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) 
! local variables...
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:)
        real, allocatable :: nxnx(:,:,:), nnx(:,:,:)
        real, allocatable :: nxnx_all_dim(:,:,:,:)
! filters for ann:
        real, allocatable :: afilt_nxnx(:,:,:,:),  afilt_nnx(:,:,:,:)
        real, allocatable :: afilt_nxnx_all_dim(:,:,:,:,:)
        real, allocatable :: x_loc(:,:), x_all(:,:)
        integer npoly,ele_type,ele, iloc,jloc,idim
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, gi, jdim
        real rnorm

        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
        allocate(nxnx(ndim,nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(x_loc(ndim,nloc)) 
        allocate(nxnx_all_dim(ndim,ndim,nloc,nloc))

!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 

        allocate(x_all(ndim,totele*nloc)) 
! 
        if(ndim==2) then
          x_all(:,1)=(/-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0/)
          x_all(:,4)=(/+1.0,+1.0/)
        endif
        if(ndim==3) then
          x_all(:,1)=(/-1.0,-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0,-1.0/)
          x_all(:,4)=(/+1.0,+1.0,-1.0/)
! 2nd layer...
          x_all(:,5)=(/-1.0,-1.0,+1.0/)
          x_all(:,6)=(/+1.0,-1.0,+1.0/)
          x_all(:,7)=(/-1.0,+1.0,+1.0/)
          x_all(:,8)=(/+1.0,+1.0,+1.0/)
        endif

! obtain 
! for filters...
        do ele = 1, totele ! VOLUME integral
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )

             nxnx=0.0
             nnx=0.0
             nxnx_all_dim=0.0
             do iloc=1,nloc
             do jloc=1,nloc
                   do idim=1,ndim
          nxnx(idim,iloc,jloc) = sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
          nnx(idim,iloc,jloc)  = sum( n(:,iloc)*nx(:,idim,jloc)*detwei(:) )
                      do jdim=1,ndim
          nxnx_all_dim(idim,jdim,iloc,jloc) = sum( nx(:,idim,iloc)*nx(:,jdim,jloc)*detwei(:) )
                      end do
                   end do
!                   do idim=1,1
!                      nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
!                   end do
             end do
             end do
        end do
! 
! now calculate filters:
        print *,'detwei:',detwei
        print *,'weight:',weight
        print *,' '
        do gi=1,ngi
          print *,'gi,idim,n(gi,:):',gi,n(gi,:)
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
!          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
!          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        print *,' '
        end do
!        stop 211
!        do iloc=1,nloc
!          print *,'iloc,nnx(1,iloc,:):',iloc,nnx(1,iloc,:)
!        end do
        allocate(afilt_nxnx(ndim,3,3,1+2*(ndim-2)), afilt_nnx(ndim,3,3,1+2*(ndim-2)) )
        allocate(afilt_nxnx_all_dim(ndim,ndim,3,3,1+2*(ndim-2)))
        afilt_nxnx=0.0
        afilt_nnx =0.0
        afilt_nxnx_all_dim=0.0
        do iiblock=1,2
        do jjblock=1,2
        do kkblock=1,ndim-1
           do ifil=1,2
           do jfil=1,2
           do kfil=1,ndim-1
!              do iidim2=1,2
!              do jjdim2=1,2
!              do kkdim2=1,ndim-1
                 iloc=1 + (2-iiblock) +(2-jjblock)*2   + (ndim-2)*(2-kkblock)*4
!                 jloc=iiblock +(jjblock-1)*2   + (kkblock-1)*4
                 jloc=ifil +(jfil-1)*2   + (kfil-1)*4
!                 jloc=iidim2 +(jjdim2-1)*2 + (kkdim2-1)*4
                 i=ifil + (iiblock-1)*1
                 j=jfil + (jjblock-1)*1
                 k=kfil + (kkblock-1)*1
                 afilt_nxnx(:,i,j,k)  = afilt_nxnx(:,i,j,k)  + nxnx(:,iloc,jloc)
                 afilt_nnx(:,i,j,k) = afilt_nnx(:,i,j,k) + nnx(:,iloc,jloc)
                 afilt_nxnx_all_dim(:,:,i,j,k) = afilt_nxnx_all_dim(:,:,i,j,k)  &
                                               + nxnx_all_dim(:,:,iloc,jloc)
!              end do
!              end do
!              end do
           end do
           end do
           end do
        end do
        end do
        end do
! 
        do idim=1,ndim
           print *,'idim=',idim
           do jloc=1,nloc
              print *,'jloc,nxnx(idim,:,jloc):',jloc,nxnx(idim,:,jloc)
           end do
        end do
! 
        do idim=1,ndim
           print *,'idim=',idim
           do k=1,1+2*(ndim-2)
           do j=1,3
              print *,'k,j,afilt_nxnx(idim,:,j,k):',k,j,afilt_nxnx(idim,:,j,k)
           end do
           end do
        end do
! 
        do idim=1,ndim
        do jdim=1,ndim
           print *,'idim,jdim=',idim,jdim
           do k=1,1+2*(ndim-2)
           do j=1,3
              print *,'k,j,afilt_nxnx_all_dim(idim,jdim,:,j,k):',k,j,afilt_nxnx_all_dim(idim,jdim,:,j,k)
           end do
           end do
        end do
        end do
!
! get rid of small values - round off error is an issue here. 
        do idim=1,ndim
           do k=1,1+2*(ndim-2)
           do j=1,3
           do i=1,3
             if( abs( afilt_nnx(idim,i,j,k))<7.e-5) afilt_nnx(idim,i,j,k)=0.0
           end do
           end do
           end do
        end do
! 
        do idim=1,ndim
           print *,' '
           do k=1,1+2*(ndim-2)
           do j=1,3
       print *,'idim,k,j,afilt_nnx(idim,:,j,k):',idim,k,j,afilt_nnx(idim,:,j,k)
           end do
           end do
        end do
! 
! normalise
        print *,' '
        print *,'normalised filter for diffusion (pre-multiplied by 1/dx^2):'
        do idim=1,ndim
           print *,'idim=',idim
           if(ndim==2) rnorm=4.0/afilt_nxnx(idim,2,2,1)
           if(ndim==3) rnorm=6.0/afilt_nxnx(idim,2,2,2)
           do k=1,1+2*(ndim-2)
           do j=1,3
              print *,'k,j,afilt_nxnx(idim,:,j,k):',k,j,afilt_nxnx(idim,:,j,k)*rnorm
           end do
           end do
           print *,'sum( abs(afilt_nxnx(idim,:,:,1:1+(ndim-2)*2))*rnorm ):',  &
                    sum( abs(afilt_nxnx(idim,:,:,1:1+(ndim-2)*2))*rnorm )
        end do ! do idim=1,ndim
! 
        print *,' '
        print *,'normalised filter for advection (pre-multiplied by 1/dx):'
        do idim=1,ndim
           if(ndim==2) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,1)))
           if(ndim==3) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,:)))
           do k=1,1+2*(ndim-2)
           do j=1,3
              print *,'idim,k,j,afilt_nnx(idim,:,j,k):', &
                       idim,k,j,afilt_nnx(idim,:,j,k)*rnorm
           end do
           end do
           print *,' '
        end do
        stop 382
        end subroutine cty_filter_orig
! 
! 
! 
! 
        subroutine set_up_cty_filter_quadratic_and_higher_fem
        implicit none
! ann filter and simple code. 
! nacross= no of nodes across in a rectangle. 
! ndim=number of dimensions=1 or 2 or 3. 
! iall_basis_funs=1 then output all the basis functions associated with the linear combination. 
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
! dxele_dim(idim)=length of element in idim direction. 
        integer nacross, ndim, iall_basis_funs 
        real, allocatable :: dxele_dim(:)  
! 
        nacross=2 ! dont use more than 5 for 2D or 3D as there is a bug. for linear fiters 
!        nacross=3 ! dont use more than 5 for 2D or 3D as there is a bug. for quadratic filters 
!        nacross=5 ! dont use more than 5 for 2D or 3D as there is a bug.
!        nacross=7 ! dont use more than 5 for 2D or 3D as there is a bug. 
        ndim=3  ! 3d 
        iall_basis_funs=0 
!        iall_basis_funs=1 
        allocate(dxele_dim(ndim))
!        dxele_dim(:)=2.0!2.0 ! dimensions 2 is the default for linear elements. 
!        dxele_dim(:)=1.0 ! dimensions 2 is the default for linear elements. 
 !       dxele_dim(:)=1.0*real(nacross-1) ! dimensions 2 is the default.  the distrance between the nodes in one direction (1.0)
        dxele_dim(1) = 1.0*real(nacross-1)    !x 
        dxele_dim(2) = 1.0*real(nacross-1)    !y 
        dxele_dim(3) = 1.0*real(nacross-1)  !z 
!        1        2       3  
!        5.0 5.0 0.05
        
!        dxele_dim(:)=1.0*real(nacross-1) ! dimensions 2 is the default. 
!        dxele_dim(:)=1.0 ! dimensions 2 is the default. 

        call cty_filter_quadratic_and_higher_fem(nacross, ndim, iall_basis_funs, dxele_dim)

        end subroutine set_up_cty_filter_quadratic_and_higher_fem
! 
! 
! 
! 
        subroutine cty_filter_quadratic_and_higher_fem(nacross, ndim, iall_basis_funs, dxele_dim)
        implicit none
! ann filter and simple code. 
! nacross= no of nodes across in a rectangle. 
! ndim=number of dimensions=1 or 2 or 3. 
! iall_basis_funs=1 then output all the basis functions associated with the linear combination. 
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
! dxele_dim(idim)=length of element in idim direction. 
        integer, intent( in ) :: nacross, ndim, iall_basis_funs      
        real, intent( in ) :: dxele_dim(ndim)    
! local variables...
!        integer totele,nloc,snloc,sngi, ngi, nface, max_face_list_no
! 2d linear:
!        integer, parameter :: totele=1,nloc=4,snloc=2,sngi=3, ngi=9, ndim=2, &
!                              nface=4, max_face_list_no=2
! 2d quadratic:
!        integer, parameter :: totele=1,nloc=9,snloc=3,sngi=4, ngi=16, ndim=2, &
!                              nface=4, max_face_list_no=2
! 2d cubic:
!        integer, parameter :: totele=1,nloc=16,snloc=4,sngi=5, ngi=25, ndim=2, &
!                              nface=4, max_face_list_no=2
!3d linear:
       integer, parameter :: totele=1,nloc=8,snloc=4,sngi=9, ngi=27, &
                              nface=6, max_face_list_no=4
!3d quadratic:
!      integer, parameter :: totele=1,nloc=27,snloc=9,sngi=16, ngi=64,  &
!                              nface=6, max_face_list_no=4
 !      integer, parameter :: totele=1,nloc=27,snloc=9,sngi=16, ngi=64, ndim=3, &
 !                             nface=6, max_face_list_no=4
! 3d cubic:
!        integer, parameter :: totele=1,nloc=64,snloc=16,sngi=25, ngi=125, ndim=3, &
!                              nface=6, max_face_list_no=4
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
! local variables...
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:), jac_inv(:,:,:)
        real, allocatable :: nxnx_all(:,:,:), nxnx(:,:,:), nnx(:,:,:), nxn(:,:,:), nn(:,:)
! filters for ann:
        real, allocatable :: afilt_nxnx_all(:,:,:,:,:), afilt_nxnx(:,:,:,:,:), afilt_nnx(:,:,:,:,:)
        real, allocatable :: afilt_resid_tran_nnx(:,:,:,:)
        real, allocatable :: afilt_nxn(:,:,:,:,:), afilt_nn(:,:,:,:)
        real, allocatable :: full_filt_nxnx_all(:,:,:,:,:,:,:) 
        real, allocatable :: full_filt_nxnx(:,:,:,:,:,:,:) 
        real, allocatable :: full_filt_nnx(:,:,:,:,:,:,:) 
        real, allocatable :: full_filt_nxn(:,:,:,:,:,:,:) 
        real, allocatable ::  full_filt_nn(:,:,:,:,:,:) 
        real, allocatable :: afilt_nxnx_all_mid(:,:,:,:) 
        real, allocatable :: afilt_nxnx_mid(:,:,:,:) 
        real, allocatable ::  afilt_nnx_mid(:,:,:,:)
        real, allocatable ::  afilt_nxn_mid(:,:,:,:)
        real, allocatable ::   afilt_nn_mid(:,:,:)
        real, allocatable :: x_loc(:,:), x_all(:,:), rsum_afilt_nxnx_overi(:)
        real, allocatable :: rcoef_mean(:), NODPOS(:),QUAPOS(:)
        integer npoly,ele_type,ele, iloc,jloc,idim,jdim
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, gi
        integer nfilt_width, ifilt_mid
        integer iacross, jacross, kacross
        integer i2,j2,k2, ifil2,jfil2,kfil2
        integer idis,jdis,kdis, ifil_dis, jfil_dis, kfil_dis 
        integer ifil2_dis, jfil2_dis, kfil2_dis 
        integer istart,ifinish, jstart,jfinish, kstart,kfinish 
        integer jrang,krang, k1,k3, ibase,nbase_print, iibase
        integer IPOLY, IQADRA, nloc2, NDNOD, jswitch, kswitch
        logical NDIFF, diff, use_base1_resid 
        real LXGP, jac, tran_value
        real SPECFU ! a function
        CHARACTER CHARA*240
        CHARACTER str_trim1*240
        CHARACTER str_trim2*240
        CHARACTER str_trim3*240
! 
        real rnorm
! 
! 2d quadratic:
 !       totele=1
 !       nloc=nacross**ndim
 !       snloc=nacross**(ndim-1)
 !       ngi=(nacross+1)**ndim
 !       sngi=(nacross+1)**(ndim-1)
!        ndim=2
  !      nface=2*ndim
 !       max_face_list_no=2**(ndim-1)

!        nacross=int( (real(nloc))**(1./real(ndim)) +0.1 )
        nfilt_width = 2*(nacross-1)+1 ! width of the filter
! 
        allocate(n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi) )
        allocate(face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no) )
        allocate(face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) )
! 
        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
        allocate(jac_inv(ndim,ndim,ngi))
        allocate(nxnx_all(ndim*ndim,nloc,nloc),nxnx(ndim,nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(nxn(ndim,nloc,nloc), nn(nloc,nloc))
        allocate(x_loc(ndim,nloc)) 
        allocate(rsum_afilt_nxnx_overi(nfilt_width))
! 
!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        if(ndim==1) then
           allocate(NODPOS(nloc),QUAPOS(ngi))
           IPOLY=1 ! Legendra
           IQADRA=1
           nloc2=2 ! BUG HERE**************
           CALL GTROOT(IPOLY,IQADRA,WEIGHT,NODPOS,QUAPOS,NGI,nloc2)
           do iloc=1,nloc ! UNIFORM SPACING 
              NODPOS(iloc)=(real(iloc-1)/real(nloc-1))*2.0 - 1.0
           end do
!       IPOLY=2 ! Cheb..
           NDIFF=.false.
           DIFF=.true.
           idim=1
           NDNOD=nloc
           print *,'just starting initial set up'

           do gi=1,ngi! Was loop 1000
          ! 
              LXGP=QUAPOS(gi)
          ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
          ! else return the Gauss-pt.
          ! 
              do iloc=1,nloc! Was loop 12000
             !
                 N(gi,ILOC) = &
                  SPECFU(NDIFF,LXGP,iloc,NDNOD,IPOLY,NODPOS)
             !
                 NLX(gi,idim,ILOC) = &
                  SPECFU(DIFF, LXGP,iloc,NDNOD,IPOLY,NODPOS)
              end do ! Was loop 12000
           end do ! Was loop 1000
        else
           print *,'going into get_shape_funs_with_faces'
           call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        endif
! 
        jswitch=min(1,max(0,ndim-1)) ! switch on 2D parts (needed for 3d)
        kswitch=min(1,max(0,ndim-2)) ! switch on 3D.

        allocate(x_all(ndim,totele*nloc)) 
! 
        
        do kacross=1,(nacross-1)*kswitch + 1
          do jacross=1,(nacross-1)*jswitch + 1
             do iacross=1,nacross
                iloc = (kacross-1)*nacross*nacross + (jacross-1)*nacross + iacross 
! dxele_dim(idim) contain the dimensions of the element...
!                x_all(1,iloc)=-1.0 + 2.*real(iacross-1)/real(nacross-1) 
                x_all(1,iloc)=-0.5*dxele_dim(1) + dxele_dim(1)*real(iacross-1)/real(nacross-1) 
  if(ndim.ge.2) x_all(2,iloc)=-0.5*dxele_dim(2) + dxele_dim(2)*real(jacross-1)/real(nacross-1)
    if(ndim==3) x_all(3,iloc)=-0.5*dxele_dim(3) + dxele_dim(3)*real(kacross-1)/real(nacross-1)
!                print *,'iloc,x_all(1,iloc):',iloc,x_all(1,iloc)
             end do
           end do
        end do
!         stop 2924

! obtain 
! for filters...
        do ele = 1, totele ! VOLUME integral
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             if(ndim==1) then
                idim=1
                do gi=1,ngi
                   jac = sum(nlx(gi,idim,:)*x_loc(idim,:))
                   jac_inv(idim,idim,gi)=1.0/jac 
                end do
                detwei(:) = weight(:)*(1./jac_inv(idim,idim,:))
                nx(:,:,:)=0.0
                do gi=1,ngi
                   do iloc=1,nloc
                      nx(gi,idim,iloc) = nx(gi,idim,iloc)+ nlx(gi,idim,iloc)*jac_inv(idim,idim,gi)
                   end do
                end do
             else
                call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )
             endif

             nxnx_all=0.0
             nxnx=0.0
             nnx=0.0
             nxn=0.0
             nn=0.0
             do iloc=1,nloc
             do jloc=1,nloc
                   do idim=1,ndim
                      do jdim=1,ndim
          nxnx_all(idim+(jdim-1)*ndim,iloc,jloc) = sum( nx(:,idim,iloc)*nx(:,jdim,jloc)*detwei(:) )
                      end do
          nxnx(idim,iloc,jloc) = sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
          nnx(idim,iloc,jloc)  = sum( n(:,iloc)*nx(:,idim,jloc)*detwei(:) )
          nxn(idim,iloc,jloc)  = sum( nx(:,idim,iloc)*n(:,jloc)*detwei(:) )
                   end do
          nn(iloc,jloc)  = sum( n(:,iloc)*n(:,jloc)*detwei(:) )
!                   do idim=1,1
!                      nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
!                   end do
             end do
             end do
        end do
! 
! now calculate filters:
        print *,'detwei:',detwei
        print *,'weight:',weight
        print *,' '
        do gi=1,ngi
          print *,'gi,idim,n(gi,:):',gi,n(gi,:)
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
!          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        end do
        print *,' '
        do idim=1,1!ndim
        do gi=1,1!ngi
!          if(idim==1) print *,'gi,idim,n(gi,:):',gi,idim,n(gi,:)
!          print *,'gi,idim,nlx(gi,idim,:):',gi,idim,nlx(gi,idim,:)
          iloc=1
          print *,'iloc,nnx(idim,iloc,:):',iloc,nnx(idim,iloc,:)
        end do
        print *,' '
        end do
!        stop 211
!        do iloc=1,nloc
!          print *,'iloc,nnx(1,iloc,:):',iloc,nnx(1,iloc,:)
!        end do
!        print *,'nacross:',nacross
!        print *,'rcoef_mean:',rcoef_mean
!        print *,'nfilt_width,nfilt_width/2:',nfilt_width,nfilt_width/2
!        stop 2921
        istart= -nfilt_width/2
        ifinish= nfilt_width/2
        jstart= 0+jswitch*istart
        jfinish=0+jswitch*ifinish
! 
        kstart=   0+kswitch*istart
        kfinish=  0+kswitch*ifinish 
        allocate(full_filt_nxnx_all(ndim*ndim, 2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish, &
                                               2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish ) )
        allocate(full_filt_nxnx(ndim, 2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish, &
                                      2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish ) )
        allocate( full_filt_nnx(ndim, 2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish, &
                                      2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish ) )
        allocate( full_filt_nxn(ndim, 2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish, &
                                      2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish ) )
        allocate( full_filt_nn(2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish, &
                               2*istart:2*ifinish, 2*jstart:2*jfinish, 2*kstart:2*kfinish ) )
!        allocate(index_nxnx(ndim, nacross,nacross,nacross, nacross,nacross,nacross) )
!        allocate( index_nnx(ndim, nacross,nacross,nacross, nacross,nacross,nacross) )
!        print *,'nacross=',nacross
!        stop 29112
        print *,'here1'
        full_filt_nxnx_all=0.0
        full_filt_nxnx    =0.0
        full_filt_nnx     =0.0
        full_filt_nxn     =0.0
        full_filt_nn      =0.0
        do kkblock=1,2*kswitch + 1*(1-kswitch)
        do jjblock=1,2*jswitch + 1*(1-jswitch)
        do iiblock=1,2
           do k=1,nacross*kswitch + 1*(1-kswitch)
           do j=1,nacross*jswitch + 1*(1-jswitch)
           do i=1,nacross
              do k2=1,nacross*kswitch + 1*(1-kswitch)
              do j2=1,nacross*jswitch + 1*(1-jswitch)
              do i2=1,nacross
                 iloc=i  +(j -1)*nacross + (k -1)*nacross*nacross
                 jloc=i2 +(j2-1)*nacross + (k2-1)*nacross*nacross
!                 print *,'iiblock,jjblock,kkblock:',iiblock,jjblock,kkblock
!                 print *,'i,j,k   :',i,j,k
!                 print *,'i2,j2,k2:',i2,j2,k2
!                 print *,'iloc,jloc:',iloc,jloc
! 
                 ifil  = i  -1 + (iiblock-2)*(nfilt_width/2)
                 jfil  =(j  -1 + (jjblock-2)*(nfilt_width/2))*jswitch
                 kfil  =(k  -1 + (kkblock-2)*(nfilt_width/2))*kswitch
! 
                 ifil2 = i2 -1 + (iiblock-2)*(nfilt_width/2)
                 jfil2 =(j2 -1 + (jjblock-2)*(nfilt_width/2))*jswitch
                 kfil2 =(k2 -1 + (kkblock-2)*(nfilt_width/2))*kswitch
! 
                 full_filt_nxnx_all(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               = full_filt_nxnx_all(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               + nxnx_all(:,iloc,jloc)
! 
                 full_filt_nxnx(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               = full_filt_nxnx(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) + nxnx(:,iloc,jloc)
! 
                  full_filt_nnx(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               =  full_filt_nnx(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) +  nnx(:,iloc,jloc)
! 
                  full_filt_nxn(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               =  full_filt_nxn(:, ifil,jfil,kfil, ifil2,jfil2,kfil2) +  nxn(:,iloc,jloc)
! 
                   full_filt_nn(   ifil,jfil,kfil, ifil2,jfil2,kfil2) &
               =   full_filt_nn(   ifil,jfil,kfil, ifil2,jfil2,kfil2) +  nn(iloc,jloc)
! 
              end do
              end do
              end do
           end do
           end do
           end do
        end do
        end do
        end do
!        print *,'full_filt_nxn:',full_filt_nnx
!        print *,'sum(abs(full_filt_nnx)):',sum(abs(full_filt_nnx))
!        print *,'nxn:',nnx
!        stop 45
        print *,'here2'
! 
! add different versions of this in a (nacross-1)**ndim cube
        allocate(afilt_nxnx_all_mid(ndim*ndim, istart:ifinish, jstart:jfinish, kstart:kfinish ) )
        allocate(afilt_nxnx_mid(ndim, istart:ifinish, jstart:jfinish, kstart:kfinish ) )
        allocate(afilt_nnx_mid(ndim,  istart:ifinish, jstart:jfinish, kstart:kfinish ) )
        allocate(afilt_nxn_mid(ndim,  istart:ifinish, jstart:jfinish, kstart:kfinish ) )
        allocate(afilt_nn_mid(        istart:ifinish, jstart:jfinish, kstart:kfinish ) )
! 
        nbase_print=1
        if(iall_basis_funs==1) then
           nbase_print=(nacross-1)**ndim
        endif
        allocate(rcoef_mean((nacross-1)**ndim))
        rcoef_mean(:)=1./real((nacross-1)**ndim) ! This is formed from the averaging.
! 
!        stop 382
        allocate(afilt_nxnx_all(ndim*ndim, nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch, nbase_print))
        allocate(afilt_nxnx(ndim, nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch, nbase_print))
        allocate( afilt_nnx(ndim, nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch, nbase_print))
        allocate( afilt_nxn(ndim, nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch, nbase_print))
        allocate( afilt_nn(nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch, nbase_print))
! 
        allocate( afilt_resid_tran_nnx(ndim, nfilt_width, 1+(nfilt_width-1)*jswitch, &
                                1+(nfilt_width-1)*kswitch))
! 
        print *,'here3'
       do ibase=1,nbase_print
        if(nbase_print>1) then
           rcoef_mean(:)    =0.0
           rcoef_mean(ibase)=1.0
        endif
! 
        afilt_nxnx_all_mid=0.0
        afilt_nxnx_mid=0.0
        afilt_nnx_mid =0.0
        afilt_nxn_mid =0.0
        afilt_nn_mid =0.0
! 
        iibase=0
! 
        do kdis=0,0 + (nacross-2)*kswitch
        do jdis=0,0 + (nacross-2)*jswitch
        do idis=0,nacross-2
           iibase=iibase+1
           do kfil2=kstart,kfinish
           do jfil2=jstart,jfinish
           do ifil2=istart,ifinish
! 
              ifil_dis = idis
              jfil_dis = jdis
              kfil_dis = kdis
! 
              ifil2_dis = ifil2 + idis
              jfil2_dis = jfil2 + jdis
              kfil2_dis = kfil2 + kdis
! 
              afilt_nxnx_all_mid(:, ifil2,jfil2,kfil2) = afilt_nxnx_all_mid(:, ifil2,jfil2,kfil2) &
              + full_filt_nxnx_all(:, ifil_dis,jfil_dis,kfil_dis, ifil2_dis,jfil2_dis,kfil2_dis) &
              * rcoef_mean(iibase)
! 
              afilt_nxnx_mid(:,ifil2,jfil2,kfil2) = afilt_nxnx_mid(:,ifil2,jfil2,kfil2) &
              + full_filt_nxnx(:,ifil_dis,jfil_dis,kfil_dis, ifil2_dis,jfil2_dis,kfil2_dis) &
              * rcoef_mean(iibase)
! 
              afilt_nnx_mid(:,ifil2,jfil2,kfil2) = afilt_nnx_mid(:,ifil2,jfil2,kfil2) &
              +  full_filt_nnx(:,ifil_dis,jfil_dis,kfil_dis, ifil2_dis,jfil2_dis,kfil2_dis) &
              * rcoef_mean(iibase)
! 
              afilt_nxn_mid(:,ifil2,jfil2,kfil2) = afilt_nxn_mid(:,ifil2,jfil2,kfil2) &
              +  full_filt_nxn(:,ifil_dis,jfil_dis,kfil_dis, ifil2_dis,jfil2_dis,kfil2_dis) &
              * rcoef_mean(iibase)
! 
              afilt_nn_mid(ifil2,jfil2,kfil2) = afilt_nn_mid(ifil2,jfil2,kfil2) &
              +  full_filt_nn(ifil_dis,jfil_dis,kfil_dis, ifil2_dis,jfil2_dis,kfil2_dis) &
              * rcoef_mean(iibase)
! 
           end do
           end do
           end do
        end do
        end do
        end do
        print *,'here3.5'

        jrang=1+(nfilt_width-1)*jswitch
        krang=1+(nfilt_width-1)*kswitch
! 
        afilt_nxnx_all(:, 1:nfilt_width, 1:jrang, 1:krang, ibase) &
          = afilt_nxnx_all_mid(:, istart:ifinish, jstart:jfinish, kstart:kfinish)
! 
        afilt_nxnx(:, 1:nfilt_width, 1:jrang, 1:krang, ibase) &
          = afilt_nxnx_mid(:, istart:ifinish, jstart:jfinish, kstart:kfinish)
! 
        afilt_nnx(:, 1:nfilt_width, 1:jrang, 1:krang, ibase) &
          = afilt_nnx_mid(:, istart:ifinish, jstart:jfinish, kstart:kfinish)
! 
        afilt_nxn(:, 1:nfilt_width, 1:jrang, 1:krang, ibase) &
          = afilt_nxn_mid(:, istart:ifinish, jstart:jfinish, kstart:kfinish)
! 
        afilt_nn(    1:nfilt_width, 1:jrang, 1:krang, ibase) &
          = afilt_nn_mid(    istart:ifinish, jstart:jfinish, kstart:kfinish)
! 
       end do ! do ibase=1,nbase_print
        print *,'here4'
!  
        use_base1_resid=.true. 
        if(use_base1_resid.and.(nbase_print>1)) then 
!        if(use_base1_resid) then 
           afilt_resid_tran_nnx(:,:,:,:)=0.0
           do ibase=1,nbase_print
              afilt_resid_tran_nnx(:,:,:,:) =  afilt_resid_tran_nnx(:,:,:,:) &
                                            + afilt_nnx(:,:,:,:,ibase) 
           end do
           afilt_resid_tran_nnx(:,:,:,:) =  afilt_resid_tran_nnx(:,:,:,:) &
                                         - afilt_nnx(:,:,:,:,1) 
           tran_value = sum(afilt_nn(:,:,:,:)) - sum(afilt_nn(:,:,:,1))
!           tran_value = sum(afilt_nn(:)) - sum(afilt_nn_bases(:,1))
           afilt_resid_tran_nnx(:,:,:,:) = afilt_resid_tran_nnx(:,:,:,:) &
                                   * sum(afilt_nn(:,:,:,:))/tran_value
! not sure where these coefficient 0.25 come from...
!           afilt_resid_tran_nnx = 0.5*(0.25*afilt_resid_tran_nnx-afilt_nnx)
           afilt_resid_tran_nnx = 0.5*0.25*afilt_resid_tran_nnx
           do ibase=1,nbase_print
              afilt_resid_tran_nnx(:,:,:,:) =  afilt_resid_tran_nnx(:,:,:,:) &
                           - 0.5*afilt_nnx(:,:,:,:,ibase)/real((nacross-1)**ndim) 
!        rcoef_mean(:)=1./real((nacross-1)**ndim) ! This is formed from the averaging.
           end do
        endif
! 
 

! 
        do idim=1,ndim
           print *,'idim=',idim
           do jloc=1,nloc
              print *,'jloc,nxnx(idim,:,jloc):',jloc,nxnx(idim,:,jloc)
           end do
        end do
! 
       do ibase=1,nbase_print
        do idim=1,ndim
           print *,'idim=',idim
           do k=1,krang
           do j=1,jrang
              print *,'k,j,afilt_nxnx(idim,:,j,k,ibase):',k,j,afilt_nxnx(idim,:,j,k,ibase)
           end do
           end do
        end do
       end do
! 
       do ibase=1,nbase_print
           do k=1,krang
           do j=1,jrang
              print *,'k,j,afilt_nn(:,j,k,ibase):',k,j,afilt_nn(:,j,k,ibase)
           end do
           end do
       end do
! 
! 
            str_trim1=chara(ndim);      k1=INDEX(str_trim1,' ')-1
            str_trim2=chara(nacross);   k2=INDEX(str_trim2,' ')-1
            str_trim3=chara(nbase_print); k3=INDEX(str_trim3,' ')-1
!            open(27, file='run/group-output-time'//str_trim(1:k)//'.csv', status='replace')
            open(27, file='afilt_ndim='//str_trim1(1:k1) &
                          //'_nacross='//str_trim2(1:k2) &
                        //'_nbase_print='//str_trim3(1:k3)//'.csv', status='replace')
! 
        write(27,*) 'Multi-dimensional convolutional filters for solving PDEs with'
        write(27,*) 'linear and higher order based methods.'
        write(27,*) 'The higher order FEM are modified FEM approaches for convolutions.'
        write(27,*) 
        write(27,*) 'afilt_nxnx_isotropic.csv:'
        write(27,*) 'The isotropic diffusion filter D^2.'
        write(27,*) 'FORTRAN output code... then the actual output of the code...'
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) '   do k=1,krang'
        write(27,*) '   do j=1,jrang'
        write(27,*) '      rsum_afilt_nxnx_overi(:)=0.0'
        write(27,*) '      do idim=1,ndim'
        write(27,*) '         rsum_afilt_nxnx_overi(:) = rsum_afilt_nxnx_overi(:) &'
        write(27,*) '                                  + afilt_nxnx(idim,:,j,k,ibase)'
        write(27,*) '      end do'
        write(27,*) '      write(27,*) rsum_afilt_nxnx_overi(:)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
           do k=1,krang 
           do j=1,jrang
              rsum_afilt_nxnx_overi(:)=0.0
              do idim=1,ndim
                 rsum_afilt_nxnx_overi(:) = rsum_afilt_nxnx_overi(:) &
                                          + afilt_nxnx(idim,:,j,k,ibase)
              end do
              write(27,*) rsum_afilt_nxnx_overi(:)
           end do
           end do
        end do
! 
        write(27,*) 
        write(27,*) 'afilt_nxnx.csv:'
        write(27,*) 'The isotropic diffusion filter D^2 but in the x-,y-, and z- directions only.'
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) 'do idim=1,ndim'
        write(27,*) '   do k=1,krang'  
        write(27,*) '   do j=1,jrang'
        write(27,*) '      write(27,*) afilt_nxnx(idim,:,j,k,ibase)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
        do idim=1,ndim
           print *,' '
           do k=1,krang
           do j=1,jrang
       print *,'idim,k,j,afilt_nxnx(idim,:,j,k,ibase):',idim,k,j,afilt_nxnx(idim,:,j,k,ibase)
              write(27,*) afilt_nxnx(idim,:,j,k,ibase)
           end do
           end do
        end do
        end do
! 
        write(27,*) 
        write(27,*) 'afilt_nxnx_all.csv:'
        write(27,*) 'The isotropic entire diffusion filter Dxx, Dxy,  Dyx, Dyy etc.' 
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) 'do idim=1,ndim'
        write(27,*) 'do jdim=1,ndim'
        write(27,*) '   do k=1,krang'  
        write(27,*) '   do j=1,jrang'
        write(27,*) '      write(27,*) afilt_nxnx_all(idim+(jdim-1)*ndim,:,j,k,ibase)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
        do idim=1,ndim
        do jdim=1,ndim
           print *,' '
           do k=1,krang
           do j=1,jrang
       print *,'idim,k,j,afilt_nxnx_all(idim+(jdim-1)*ndim,:,j,k,ibase):', &
                idim,k,j,afilt_nxnx_all(idim+(jdim-1)*ndim,:,j,k,ibase)
              write(27,*) afilt_nxnx_all(idim+(jdim-1)*ndim,:,j,k,ibase)
           end do
           end do
        end do
        end do
        end do
! 
! get rid of small values - round off error is an issue here. 
        write(27,*) 
        write(27,*) 'afilt_nnx.csv:'
        write(27,*) 'The gradient (advection) filters in the x-, y-, z- directions' 
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) 'do idim=1,ndim'
        write(27,*) '   do k=1,krang'  
        write(27,*) '   do j=1,jrang'
        write(27,*) '      write(27,*) afilt_nnx(idim,:,j,k,ibase)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
        do idim=1,ndim
           do k=1,krang
           do j=1,jrang
           do i=1,nfilt_width
             if( abs( afilt_nnx(idim,i,j,k,ibase))<7.e-10) afilt_nnx(idim,i,j,k,ibase)=0.0
           end do
              write(27,*) afilt_nnx(idim,:,j,k,ibase)
           end do
           end do
        end do
        end do
! 
! get rid of small values - round off error is an issue here. 
        write(27,*) 
        write(27,*) 'afilt_nxn.csv:'
        write(27,*) 'The conservative gradient (advection) filters in the x-, y-, z- directions' 
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) 'do idim=1,ndim'
        write(27,*) '   do k=1,krang'  
        write(27,*) '   do j=1,jrang'
        write(27,*) '      write(27,*) afilt_nxn(idim,:,j,k,ibase)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
        do idim=1,ndim
           do k=1,krang
           do j=1,jrang
           do i=1,nfilt_width
             if( abs( afilt_nxn(idim,i,j,k,ibase))<7.e-10) afilt_nxn(idim,i,j,k,ibase)=0.0
           end do
              write(27,*) afilt_nxn(idim,:,j,k,ibase)
           end do
           end do
        end do
        end do
! 
        write(27,*) 
        write(27,*) 'afilt_nn.csv:'
        write(27,*) 'The consistent mass matrix filter.' 
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do ibase=1,nbase_print' 
        write(27,*) 'do k=1,krang'
        write(27,*) 'do j=1,jrang'
        write(27,*) '   write(27,*) afilt_nn(:,j,k)'
        write(27,*) 'end do'
        write(27,*) 'end do'
        write(27,*) 'end do'
        do ibase=1,nbase_print
           do k=1,krang
           do j=1,jrang
              write(27,*) afilt_nn(:,j,k,ibase)
           end do
           end do
        end do
! 
        write(27,*) 
        write(27,*) 'afilt_ml.csv:'
        write(27,*) 'The lumped mass term (scalar for this node/cell).'
        write(27,*) 'do ibase=1,nbase_print'
        write(27,*) '   write(27,*) sum(afilt_nn(:,:,:,ibase))'
        write(27,*) 'end do'
        do ibase=1,nbase_print
              write(27,*) sum(afilt_nn(:,:,:,ibase))
        end do
! 
        write(27,*) 
        write(27,*) 'afilt_ml_check.csv:'
        write(27,*) 'The lumped mass term (scalar for this node/cell).'
        write(27,*) 'write(27,*) (2./real(nacross-1))**ndim'
              write(27,*) (2./real(nacross-1))**ndim
! 
        write(27,*) 
        write(27,*) 'dxele_dim.csv:'
        write(27,*) 'The dimensions of the hexahedra element.'
        write(27,*) 'write(27,*) dxele_dim(:)'
              write(27,*) dxele_dim(:)
! 
        write(27,*) 
        write(27,*) 'integer_options_used.csv:'
        write(27,*) 'write(27,*) nacross, ndim, iall_basis_funs, nbase_print'
        write(27,*) nacross, ndim, iall_basis_funs, nbase_print
! 
! afilt_resid_tran_nnx...
! get rid of small values - round off error is an issue here. 
        if(use_base1_resid.and.(nbase_print>1)) then 
        write(27,*) 
        write(27,*) 'afilt_resid_tran_nnx.csv:'
        write(27,*) 'The residual gradient error (advection) filters in the x-, y-, z- directions' 
        write(27,*) 'jrang=1+(nfilt_width-1)*jswitch'
        write(27,*) 'krang=1+(nfilt_width-1)*kswitch'
        write(27,*) 'do idim=1,ndim'
        write(27,*) '   do k=1,krang'  
        write(27,*) '   do j=1,jrang'
        write(27,*) '      write(27,*) afilt_resid_tran_nnx(idim,:,j,k)'
        write(27,*) '   end do'
        write(27,*) '   end do'
        write(27,*) 'end do'
        do idim=1,ndim
           do k=1,krang
           do j=1,jrang
           do i=1,nfilt_width
             if( abs( afilt_resid_tran_nnx(idim,i,j,k))<7.e-10) afilt_resid_tran_nnx(idim,i,j,k)=0.0
           end do
              write(27,*) afilt_resid_tran_nnx(idim,:,j,k)
           end do
           end do
        end do 
        endif ! if(use_base1_resid.and.(nbase_print>1)) then
        close(27)  
! 
! normalise
        print *,' '
        print *,'normalised filter for diffusion components (pre-multiplied by 1/dx^2):'
           ifilt_mid=1 + nfilt_width/2
        do ibase=1,nbase_print
        do idim=1,ndim
           print *,'idim=',idim
           if(ndim==1) rnorm=2.0/afilt_nxnx(idim,ifilt_mid,1,1,ibase)
           if(ndim==2) rnorm=4.0/afilt_nxnx(idim,ifilt_mid,ifilt_mid,1,ibase)
           if(ndim==3) rnorm=6.0/afilt_nxnx(idim,ifilt_mid,ifilt_mid,ifilt_mid,ibase)
           do k=1,krang
           do j=1,jrang
              print *,'k,j,afilt_nxnx(idim,:,j,k,ibase):', &
                       k,j,afilt_nxnx(idim,:,j,k,ibase)*rnorm
           end do
           end do
           print *,'sum( abs(afilt_nxnx(idim,:,:,1:1+(ndim-2)*2,ibase)*rnorm )):',  &
                    sum( abs(afilt_nxnx(idim,:,:,1:1+(ndim-2)*2,ibase)*rnorm ))
        end do ! do idim=1,ndim
        end do ! do ibase=1,nbase_print
! 
        print *,'normalised filter for diffusion (pre-multiplied by 1/dx^2):'
           ifilt_mid=1 + nfilt_width/2
        do ibase=1,nbase_print
           do k=1,krang
           do j=1,jrang
              rsum_afilt_nxnx_overi(:)=0.0
              do idim=1,ndim
                 if(ndim==1) rnorm=2.0/afilt_nxnx(idim,ifilt_mid,1,1,ibase)
                 if(ndim==2) rnorm=4.0/afilt_nxnx(idim,ifilt_mid,ifilt_mid,1,ibase)
                 if(ndim==3) rnorm=6.0/afilt_nxnx(idim,ifilt_mid,ifilt_mid,ifilt_mid,ibase)
                 rsum_afilt_nxnx_overi(:) = rsum_afilt_nxnx_overi(:) &
                                          + afilt_nxnx(idim,:,j,k,ibase)*rnorm
              end do
              print *,'k,j,afilt_nxnx(sum(idim),:,j,k,ibase):',k,j,rsum_afilt_nxnx_overi(:)
           end do
           end do
        end do
! 
        print *,' '
        print *,'normalised filter for advection (pre-multiplied by 1/dx):'
        do ibase=1,nbase_print
        do idim=1,ndim
           if(ndim==1) rnorm=1.0/sum(abs(afilt_nnx(idim,:,1,1,ibase)))
           if(ndim==2) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,1,ibase)))
           if(ndim==3) rnorm=1.0/sum(abs(afilt_nnx(idim,:,:,:,ibase)))
           do k=1,krang
           do j=1,jrang
!              print *,'idim,k,j:',idim,k,j
!              print *,'rnorm:',rnorm
              print *,'idim,k,j,afilt_nnx(idim,:,j,k,ibase):', &
                       idim,k,j,afilt_nnx(idim,:,j,k,ibase)*rnorm
           end do
           end do
           print *,' '
        end do
        end do
! 
        stop 382
        end subroutine cty_filter_quadratic_and_higher_fem
! 
! 
! 
! 
      CHARACTER*240 FUNCTION CHARA(CURPRO)
!     CURPRO must be the same in calling subroutine.
      INTEGER MXNLEN
      PARAMETER(MXNLEN=5)
      INTEGER DIGIT(MXNLEN),II,J,TENPOW,CURPRO,K
      CHARACTER STRI*240
      CHARACTER*1 CHADIG(MXNLEN)
      LOGICAL SWITCH
!     MXNLEN=max no of digits in integer I.
!     This function converts integer I to a string.
      II=CURPRO
      SWITCH=.FALSE.
      K=0
      STRI=' '
      do  J=MXNLEN,1,-1! Was loop 10
         TENPOW=10**(J-1)
         DIGIT(J)=II/TENPOW
         II=II-DIGIT(J)* TENPOW
!     ewrite(3,*)'DIGIT*J)=',DIGIT(J),' TENOW=',TENPOW
!     STRI=STRI//CHAR(DIGIT(J)+48)
         IF(DIGIT(J).EQ.0) CHADIG(J)='0'
         IF(DIGIT(J).EQ.1) CHADIG(J)='1'
         IF(DIGIT(J).EQ.2) CHADIG(J)='2'
         IF(DIGIT(J).EQ.3) CHADIG(J)='3'
         IF(DIGIT(J).EQ.4) CHADIG(J)='4'
         IF(DIGIT(J).EQ.5) CHADIG(J)='5'
         IF(DIGIT(J).EQ.6) CHADIG(J)='6'
         IF(DIGIT(J).EQ.7) CHADIG(J)='7'
         IF(DIGIT(J).EQ.8) CHADIG(J)='8'
         IF(DIGIT(J).EQ.9) CHADIG(J)='9'
!         Q=INDEX(STRI,' ')-1
!         IF(Q.LT.1) STRI=CHADIG
!         IF(Q.GE.1) STRI=STRI(1:Q)//CHADIG
!         IF(DIGIT(J).NE.0) SWITCH=.TRUE.
!         IF(SWITCH) K=K+1
      end do ! Was loop 10
       k=0
      do  J=MXNLEN,1,-1! Was loop 20
         IF(SWITCH) THEN 
           STRI=STRI(1:K)//CHADIG(J)
           K=K+1
         ENDIF
         IF((CHADIG(J).NE.'0').AND.(.NOT.SWITCH)) THEN 
           SWITCH=.TRUE.
           STRI=CHADIG(J)
           K=1
         ENDIF
      end do ! Was loop 20
       IF(K.EQ.0) THEN 
         STRI='0'
          K=1
        ENDIF
!       ewrite(3,*)'STRI=',STRI,' K=',K
!       ewrite(3,*)'STRI(MXNLEN-K+1:MXNLEN)=',STRI(MXNLEN-K+2:MXNLEN+1)
!       CHARA=STRI(MXNLEN-K+2:MXNLEN+1)
        CHARA=STRI(1:K)
       END
! 
! 
! 
! 
! Python interface is: 
! psi_no_bc = sfc_solver_it_3(psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
!                                 nonods_sfc_all_grids, nlevel, &
!                                 a_no_bc, b_no_bc, relax, &
!                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
!                                 ncola_no_bc, nonods_no_bc, &
!                                 nfilt_size_sfc)  
        subroutine sfc_solver_it_3(psi_no_bc, psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 nfilt_size_sfc)  
! Solve the system a_no_bc*psi_no_bc = b_no_bc with a single multi-grid iteration and 1 SFC. 
! it uses a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. 
! It does this with a kernal size of 3. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! if(iscale_matrices==1) then assume the matricies have been divided through by mass matrix.
! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont.
! i_jacobi_full_matrix=1 then include Jacobi relaxation on full matrix, =0 dont.
! Default values: iscale_matrices=0, i_jacobi_on_finest_sfc=0, i_jacobi_full_matrix=1
! one must have either i_jacobi_on_finest_sfc=1 and/or i_jacobi_full_matrix=1
! 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola_no_bc, nonods_no_bc, nonods_sfc_all_grids, nlevel
        integer, intent( in ) :: iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer, intent( in ) :: nfilt_size_sfc
        real, intent( out ) :: psi_no_bc(nonods_no_bc)
        real, intent( in ) :: a_sfc(nfilt_size_sfc,nonods_sfc_all_grids) 
        real, intent( in ) :: psi_no_bc_guess(nonods_no_bc), relax(nlevel) 
        real, intent( in ) :: a_no_bc(ncola_no_bc), b_no_bc(nonods_no_bc)
        integer, intent( in ) :: fin_sfc_nonods(nlevel+1)
        integer, intent( in ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola_no_bc)
        integer, intent( in ) :: sfc_node_ordering(nonods_no_bc)
! local variables...
        real, allocatable :: resid_no_bc(:), resid_sfc_all(:), rhs_sfc_all(:)
        real, allocatable :: delta_psi_temp(:),delta_psi(:),delta_psi_sfc(:)
        real rr
        integer inod, count, jnod, inod_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer inod_sfc_all, ipt, inod_sfc_fine, inod_sfc_fine_all,inod_sfc_fine_all2
        integer ilevel_finish
! 
! calculate nlevel from nonods
        print *,'just inside sfc_solver_it_3'
        if(nfilt_size_sfc.ne.3) stop 2829
!        stop 2922
        allocate(resid_no_bc(nonods_no_bc), resid_sfc_all(nonods_sfc_all_grids))
        allocate(delta_psi_sfc(nonods_sfc_all_grids), delta_psi(nonods_no_bc))
        allocate(rhs_sfc_all(nonods_sfc_all_grids), delta_psi_temp(nonods_no_bc))
! 
        psi_no_bc = psi_no_bc_guess ! need to do this because of python interface (can not have an inout variable)
        resid_no_bc=b_no_bc
        do inod=1,nonods_no_bc
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
!              print *,'inod,jnod,count,a_no_bc(count):',inod,jnod,count,a_no_bc(count)
              resid_no_bc(inod)=resid_no_bc(inod)-a_no_bc(count)*psi_no_bc(jnod) 
           end do
!           stop 7
        end do
!        stop 229
! map to sfc ordering
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           resid_sfc_all(inod_sfc) = resid_no_bc(inod)
        end do
! 
! coarsen the residual...
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         resid_sfc_all(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         resid_sfc_all(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           if(iscale_matrices==1) then ! assume the matricies have been divided through by mass matrix...
              ipt=fin_sfc_nonods(ilevel)
              resid_sfc_all(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
           endif
!           print *,'ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1):', &
!                    ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
        end do
!        stop 227
! 
! perform Jacobi relaxation on the 1d SFC and on each SFC grid level up to the 2nd. 
        delta_psi_sfc = 0.0
        rhs_sfc_all = 0.0
! Jacobi relaxation with one cell...
        ilevel=nlevel
        do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
           rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           delta_psi_sfc(inod_sfc_all)  = &
                        relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc(2,inod_sfc_all) &
                                        +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
        end do
! map to finer grid...
        sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
        sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
        call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
! 
! Jacobi relaxation on the other grids...
        ilevel_finish=2-i_jacobi_on_finest_sfc ! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont
        do ilevel=nlevel-1,ilevel_finish,-1
!        do ilevel=nlevel-1,1,-1
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           end do
           inod_sfc_all=fin_sfc_nonods(ilevel)
           rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all)    &
                                    -1.0*a_sfc(3,inod_sfc_all) * delta_psi_sfc(inod_sfc_all+1)
           do inod_sfc_all=fin_sfc_nonods(ilevel) +1,fin_sfc_nonods(ilevel+1)-1 -1
              rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -1.0*a_sfc(1,inod_sfc_all) * delta_psi_sfc(inod_sfc_all-1) &
                                    -1.0*a_sfc(3,inod_sfc_all) * delta_psi_sfc(inod_sfc_all+1) 
           end do
           inod_sfc_all=fin_sfc_nonods(ilevel+1)-1
           rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all)  &
                                    -1.0*a_sfc(1,inod_sfc_all) * delta_psi_sfc(inod_sfc_all-1)
! Jacobi relaxation...
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              delta_psi_sfc(inod_sfc_all)  = &
                     relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc(2,inod_sfc_all) &
                                           +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
           end do
! map to finer grid
   if(ilevel.ne.1) then
           sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
           sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
           call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
   endif
        end do ! do ilevel=nlevel-1,2,-1
!        ilevel=1
!        print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1):', &
!                 fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)
!        sfc_nonods_fine = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
!        print *,'delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel)-1+sfc_nonods_fine):', &
!                 delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel)-1+sfc_nonods_fine)
!        stop 29211
! 
! For ilevel=1 use Jacobi iteration with the original matrix...
        ilevel=1
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           delta_psi_temp(inod)=delta_psi_sfc(inod_sfc) 
        end do
        delta_psi=delta_psi_temp
     if(i_jacobi_full_matrix==1) then ! include Jacobi relaxation on full matrix.
        do inod=1,nonods_no_bc
           rr=resid_no_bc(inod)
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              rr = rr - a_no_bc(count)*delta_psi_temp(jnod) 
           end do
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              if(jnod==inod) then ! diagonal...
                 rr = rr + a_no_bc(count)*delta_psi_temp(jnod) 
                 delta_psi(inod) = relax(ilevel) * rr /a_no_bc(count) &
                                   +(relax(ilevel)-1.0) * delta_psi_temp(inod)
              endif
           end do
        end do
      endif
        psi_no_bc = psi_no_bc + delta_psi
        print *,'just leaving sfc_solver_it_3'
!        stop 2921
        
        return 
        end subroutine sfc_solver_it_3
! 
! 
! 
! 
! Python interface is: 
! psi_no_bc = sfc_solver_it_n(psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
!                                 nonods_sfc_all_grids, nlevel, &
!                                 a_no_bc, b_no_bc, relax, &
!                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
!                                 ncola_no_bc, nonods_no_bc, &
!                                 nfilt_size_sfc)  
        subroutine sfc_solver_it_n(psi_no_bc, psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 nfilt_size_sfc)  
! Solve the system a_no_bc*psi_no_bc = b_no_bc with a single multi-grid iteration and 1 SFC. 
! it uses a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. 
! It does this with a kernal size of nfilt_size_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! if(iscale_matrices==1) then assume the matricies have been divided through by mass matrix.
! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont.
! i_jacobi_full_matrix=1 then include Jacobi relaxation on full matrix, =0 dont.
! Default values: iscale_matrices=0, i_jacobi_on_finest_sfc=0, i_jacobi_full_matrix=1
! one must have either i_jacobi_on_finest_sfc=1 and/or i_jacobi_full_matrix=1
! 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola_no_bc, nonods_no_bc, nonods_sfc_all_grids, nlevel
        integer, intent( in ) :: iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer, intent( in ) :: nfilt_size_sfc
        real, intent( out ) :: psi_no_bc(nonods_no_bc)
        real, intent( in ) :: a_sfc(nfilt_size_sfc,nonods_sfc_all_grids) 
        real, intent( in ) :: psi_no_bc_guess(nonods_no_bc), relax(nlevel) 
        real, intent( in ) :: a_no_bc(ncola_no_bc), b_no_bc(nonods_no_bc)
        integer, intent( in ) :: fin_sfc_nonods(nlevel+1)
        integer, intent( in ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola_no_bc)
        integer, intent( in ) :: sfc_node_ordering(nonods_no_bc)
! local variables...
        real, allocatable :: resid_no_bc(:), resid_sfc_all(:), rhs_sfc_all(:)
        real, allocatable :: delta_psi_temp(:),delta_psi(:),delta_psi_sfc(:)
        real rr
        integer inod, count, jnod, inod_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer inod_sfc_all, ipt, inod_sfc_fine, inod_sfc_fine_all,inod_sfc_fine_all2
        integer ilevel_finish, inod_sfc_all_start, inod_sfc_all_finish
        integer modified_inod_sfc_all_start, modified_inod_sfc_all_finish
        integer ifilt_diag, ifilt, icent_ifilt
! 
! calculate nlevel from nonods
        print *,'just inside sfc_solver_it_n'
!        stop 2922
        allocate(resid_no_bc(nonods_no_bc), resid_sfc_all(nonods_sfc_all_grids))
        allocate(delta_psi_sfc(nonods_sfc_all_grids), delta_psi(nonods_no_bc))
        allocate(rhs_sfc_all(nonods_sfc_all_grids), delta_psi_temp(nonods_no_bc))
! 
        ifilt_diag = 1+nfilt_size_sfc/2 
! 
        psi_no_bc = psi_no_bc_guess ! need to do this because of python interface (can not have an inout variable)
        resid_no_bc=b_no_bc
        do inod=1,nonods_no_bc
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
!              print *,'inod,jnod,count,a_no_bc(count):',inod,jnod,count,a_no_bc(count)
              resid_no_bc(inod)=resid_no_bc(inod)-a_no_bc(count)*psi_no_bc(jnod) 
           end do
!           stop 7
        end do
!        stop 229
! map to sfc ordering
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           resid_sfc_all(inod_sfc) = resid_no_bc(inod)
        end do
! 
! coarsen the residual...
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         resid_sfc_all(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         resid_sfc_all(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           if(iscale_matrices==1) then ! assume the matricies have been divided through by mass matrix...
              ipt=fin_sfc_nonods(ilevel)
              resid_sfc_all(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
           endif
!           print *,'ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1):', &
!                    ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
        end do
!        stop 227
! 
! perform Jacobi relaxation on the 1d SFC and on each SFC grid level up to the 2nd or 1st. 
        delta_psi_sfc = 0.0
        rhs_sfc_all = 0.0
!         print *,'ifilt_diag:',ifilt_diag
! 
! Jacobi relaxation on the all grids...
        ilevel_finish=2-i_jacobi_on_finest_sfc ! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont
        do ilevel=nlevel,ilevel_finish,-1
!           print *,'ilevel=',ilevel
!        do ilevel=nlevel-1,1,-1
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
!           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           end do
!           print *,'h1'
           do ifilt = 1, nfilt_size_sfc
              icent_ifilt = ifilt  -  ifilt_diag
              modified_inod_sfc_all_start  = &
                         max(inod_sfc_all_start  - icent_ifilt,inod_sfc_all_start)
              modified_inod_sfc_all_finish = &
                         min(inod_sfc_all_finish - icent_ifilt,inod_sfc_all_finish)
              do inod_sfc_all = modified_inod_sfc_all_start, modified_inod_sfc_all_finish
!              do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
!            if((inod_sfc_all+icent_ifilt.ge.inod_sfc_all_start) &
!          .and.(inod_sfc_all+icent_ifilt.le.inod_sfc_all_finish)) then
                 rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -a_sfc(ifilt,inod_sfc_all) * delta_psi_sfc(inod_sfc_all+icent_ifilt) 
!            else
!               print *,'problem here'
!                stop 2929
!            endif
              end do  
           end do    
!           print *,'h2'
! we have included the diagonal so take it away again...
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish    
              rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    +a_sfc(ifilt_diag,inod_sfc_all) * delta_psi_sfc(inod_sfc_all)
!              the_elthorne_variable   = 7e7*a_sfc(imid_ifilt,inod_sfc_all) * delta_psi_sfc(inod_sfc_all)
           end do
! Jacobi relaxation...
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              delta_psi_sfc(inod_sfc_all)  = &
                     relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc(ifilt_diag,inod_sfc_all) &
                                           +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
           end do
! map to finer grid
   if(ilevel.ne.1) then
           sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
           sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
           call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
   endif
        end do ! do ilevel=nlevel,ilevel_finish,-1
!        ilevel=1
!        print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1):', &
!                 fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)
!        sfc_nonods_fine = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
!        print *,'delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel)-1+sfc_nonods_fine):', &
!                 delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel)-1+sfc_nonods_fine)
!        stop 29211
! 
! For ilevel=1 use Jacobi iteration with the original matrix...
        ilevel=1
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           delta_psi_temp(inod)=delta_psi_sfc(inod_sfc) 
        end do
        delta_psi=delta_psi_temp
     if(i_jacobi_full_matrix==1) then ! include Jacobi relaxation on full matrix.
        do inod=1,nonods_no_bc
           rr=resid_no_bc(inod)
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              rr = rr - a_no_bc(count)*delta_psi_temp(jnod) 
           end do
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              if(jnod==inod) then ! diagonal...
                 rr = rr + a_no_bc(count)*delta_psi_temp(jnod) 
                 delta_psi(inod) = relax(ilevel) * rr /a_no_bc(count) &
                                   +(relax(ilevel)-1.0) * delta_psi_temp(inod)
              endif
           end do
        end do
      endif
        psi_no_bc = psi_no_bc + delta_psi
        print *,'just leaving sfc_solver_it_n'
!        stop 2921
        
        return 
        end subroutine sfc_solver_it_n
! 
! 
! 
! 
! Python interface is: 
! psi_no_bc = sfc_solver_it_unstructured(psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
!                                 nonods_sfc_all_grids, nlevel, &
!                                 a_no_bc, b_no_bc, relax, &
!                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
!                                 ncola_no_bc, nonods_no_bc, &
!                                 nfilt_size_sfc)  
        subroutine sfc_solver_it_unstructured(psi_no_bc, psi_no_bc_guess, a_sfc_all_un, &
                                 fina_sfc_all_un, cola_sfc_all_un, &
                                 ncola_sfc_all_un, fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix)  
! Solve the system a_no_bc*psi_no_bc = b_no_bc with a single multi-grid iteration and 1 SFC. 
! it uses a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. 
! It does this with a kernal size of nfilt_size_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! if(iscale_matrices==1) then assume the matricies have been divided through by mass matrix.
! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont.
! i_jacobi_full_matrix=1 then include Jacobi relaxation on full matrix, =0 dont.
! Default values: iscale_matrices=0, i_jacobi_on_finest_sfc=0, i_jacobi_full_matrix=1
! one must have either i_jacobi_on_finest_sfc=1 and/or i_jacobi_full_matrix=1
! 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola_no_bc, nonods_no_bc, nonods_sfc_all_grids, nlevel
        integer, intent( in ) :: iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer, intent( in ) :: ncola_sfc_all_un
        real, intent( out ) :: psi_no_bc(nonods_no_bc)
        real, intent( in ) :: a_sfc_all_un(ncola_sfc_all_un) 
        real, intent( in ) :: psi_no_bc_guess(nonods_no_bc), relax(nlevel) 
        real, intent( in ) :: a_no_bc(ncola_no_bc), b_no_bc(nonods_no_bc)
        integer, intent( in ) :: fin_sfc_nonods(nlevel+1)
        integer, intent( in ) :: fina_sfc_all_un(nonods_sfc_all_grids+1)
        integer, intent( in ) :: cola_sfc_all_un(ncola_sfc_all_un)
        integer, intent( in ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola_no_bc)
        integer, intent( in ) :: sfc_node_ordering(nonods_no_bc)
! local variables...
        real, allocatable :: resid_no_bc(:), resid_sfc_all(:), rhs_sfc_all(:)
        real, allocatable :: delta_psi_temp(:),delta_psi(:),delta_psi_sfc(:)
        integer, allocatable :: mida_sfc_all_un(:)
        real rr
        integer inod, count, jnod, inod_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer inod_sfc_all, ipt, inod_sfc_fine, inod_sfc_fine_all,inod_sfc_fine_all2
        integer ilevel_finish, inod_sfc_all_start, inod_sfc_all_finish
        integer jcol_sfc_all, jnod_sfc_all
! 
! calculate nlevel from nonods
        print *,'just inside sfc_solver_it_unstructured'
!        stop 2922
        allocate(resid_no_bc(nonods_no_bc), resid_sfc_all(nonods_sfc_all_grids))
        allocate(delta_psi_sfc(nonods_sfc_all_grids), delta_psi(nonods_no_bc))
        allocate(rhs_sfc_all(nonods_sfc_all_grids), delta_psi_temp(nonods_no_bc))
        allocate(mida_sfc_all_un(nonods_sfc_all_grids))
! 
! find the diagonal...
        do inod_sfc_all = 1, nonods_sfc_all_grids  
           do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
              jnod_sfc_all=cola_sfc_all_un(count)
              if(jnod_sfc_all==inod_sfc_all) mida_sfc_all_un(inod_sfc_all)=count
           end do
        end do
! 
        psi_no_bc = psi_no_bc_guess ! need to do this because of python interface (can not have an inout variable)
        resid_no_bc=b_no_bc
        do inod=1,nonods_no_bc
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
!              print *,'inod,jnod,count,a_no_bc(count):',inod,jnod,count,a_no_bc(count)
              resid_no_bc(inod)=resid_no_bc(inod)-a_no_bc(count)*psi_no_bc(jnod) 
           end do
!           stop 7
        end do
!        stop 229
! map to sfc ordering
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           resid_sfc_all(inod_sfc) = resid_no_bc(inod)
        end do
! 
! coarsen the residual...
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         resid_sfc_all(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         resid_sfc_all(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           if(iscale_matrices==1) then ! assume the matricies have been divided through by mass matrix...
              ipt=fin_sfc_nonods(ilevel)
              resid_sfc_all(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
           endif
!           print *,'ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1):', &
!                    ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
        end do
!        stop 227
! 
! perform Jacobi relaxation on the 1d SFC and on each SFC grid level up to the 2nd or 1st. 
        delta_psi_sfc = 0.0
        rhs_sfc_all = 0.0
!         print *,'ifilt_diag:',ifilt_diag
! 
! Jacobi relaxation on the all grids...
        ilevel_finish=2-i_jacobi_on_finest_sfc ! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont
        do ilevel=nlevel,ilevel_finish,-1
!           print *,'ilevel=',ilevel
!        do ilevel=nlevel-1,1,-1
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           end do
!           print *,'h1'
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc(jcol_sfc_all) 
              end do  
           end do    
!           print *,'h2'
! we have included the diagonal so take it away again...
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish    
              rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    +a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) * delta_psi_sfc(inod_sfc_all)
!              the_elthorne_variable   = 7e7*a_sfc(imid_ifilt,inod_sfc_all) * delta_psi_sfc(inod_sfc_all)
           end do
! Jacobi relaxation...
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              delta_psi_sfc(inod_sfc_all)  = &
                     relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) &
                                           +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
           end do
! map to finer grid
   if(ilevel.ne.1) then
           sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
           sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
           call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
   endif
        end do ! do ilevel=nlevel,ilevel_finish,-1
! 
! For ilevel=1 use Jacobi iteration with the original matrix...
        ilevel=1
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           delta_psi_temp(inod)=delta_psi_sfc(inod_sfc) 
        end do
        delta_psi=delta_psi_temp
     if(i_jacobi_full_matrix==1) then ! include Jacobi relaxation on full matrix.
        do inod=1,nonods_no_bc
           rr=resid_no_bc(inod)
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              rr = rr - a_no_bc(count)*delta_psi_temp(jnod) 
           end do
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              if(jnod==inod) then ! diagonal...
                 rr = rr + a_no_bc(count)*delta_psi_temp(jnod) 
                 delta_psi(inod) = relax(ilevel) * rr /a_no_bc(count) &
                                   +(relax(ilevel)-1.0) * delta_psi_temp(inod)
              endif
           end do
        end do
      endif
        psi_no_bc = psi_no_bc + delta_psi
        print *,'just leaving sfc_solver_it_unstructured'
!        stop 2921
        
        return 
        end subroutine sfc_solver_it_unstructured
! 
! 
! 
! Python interface is: 
! psi_no_bc = sfc_solver_it_unstructured_fcycle(psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
!                                 nonods_sfc_all_grids, nlevel, &
!                                 a_no_bc, b_no_bc, relax, &
!                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
!                                 ncola_no_bc, nonods_no_bc, &
!                                 nfilt_size_sfc)  
        subroutine sfc_solver_it_unstructured_fcycle(psi_no_bc, psi_no_bc_guess, a_sfc_all_un, &
                                 fina_sfc_all_un, cola_sfc_all_un, &
                                 ncola_sfc_all_un, fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax, &
                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix, &
                                 top_level)  
! Solve the system a_no_bc*psi_no_bc = b_no_bc with a single multi-grid iteration and 1 SFC. 
! it uses a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. 
! It does this with a kernal size of nfilt_size_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! if(iscale_matrices==1) then assume the matricies have been divided through by mass matrix.
! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont.
! i_jacobi_full_matrix=1 then include Jacobi relaxation on full matrix, =0 dont.
! Default values: iscale_matrices=0, i_jacobi_on_finest_sfc=0, i_jacobi_full_matrix=1
! one must have either i_jacobi_on_finest_sfc=1 and/or i_jacobi_full_matrix=1
! 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola_no_bc, nonods_no_bc, nonods_sfc_all_grids, nlevel
        integer, intent( in ) :: iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer, intent( in ) :: top_level
        integer, intent( in ) :: ncola_sfc_all_un
        real, intent( out ) :: psi_no_bc(nonods_no_bc)
        real, intent( in ) :: a_sfc_all_un(ncola_sfc_all_un) 
        real, intent( in ) :: psi_no_bc_guess(nonods_no_bc), relax(nlevel) 
        real, intent( in ) :: a_no_bc(ncola_no_bc), b_no_bc(nonods_no_bc)
        integer, intent( in ) :: fin_sfc_nonods(nlevel+1)
        integer, intent( in ) :: fina_sfc_all_un(nonods_sfc_all_grids+1)
        integer, intent( in ) :: cola_sfc_all_un(ncola_sfc_all_un)
        integer, intent( in ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola_no_bc)
        integer, intent( in ) :: sfc_node_ordering(nonods_no_bc)
! local variables...
        real, allocatable :: resid_no_bc(:), resid_sfc_all(:), rhs_sfc_all(:)
        real, allocatable :: delta_psi_temp(:),delta_psi(:),delta_psi_sfc(:)
        integer, allocatable :: mida_sfc_all_un(:)
        real rr
        integer inod, count, jnod, inod_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer inod_sfc_all, ipt, inod_sfc_fine, inod_sfc_fine_all,inod_sfc_fine_all2
        integer ilevel_finish, inod_sfc_all_start, inod_sfc_all_finish
        integer jcol_sfc_all, jnod_sfc_all, icycle,ncycle,icycle_start,icycle_top
! 
! calculate nlevel from nonods
        print *,'just inside sfc_solver_it_unstructured_fcycle'
!        stop 2922
        allocate(resid_no_bc(nonods_no_bc), resid_sfc_all(nonods_sfc_all_grids))
        allocate(delta_psi_sfc(nonods_sfc_all_grids), delta_psi(nonods_no_bc))
        allocate(rhs_sfc_all(nonods_sfc_all_grids), delta_psi_temp(nonods_no_bc))
        allocate(mida_sfc_all_un(nonods_sfc_all_grids))
! 
! find the diagonal...
        do inod_sfc_all = 1, nonods_sfc_all_grids  
           do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
              jnod_sfc_all=cola_sfc_all_un(count)
              if(jnod_sfc_all==inod_sfc_all) mida_sfc_all_un(inod_sfc_all)=count
           end do
        end do
! 
        psi_no_bc = psi_no_bc_guess ! need to do this because of python interface (can not have an inout variable)
        resid_no_bc=b_no_bc
        do inod=1,nonods_no_bc
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
!              print *,'inod,jnod,count,a_no_bc(count):',inod,jnod,count,a_no_bc(count)
              resid_no_bc(inod)=resid_no_bc(inod)-a_no_bc(count)*psi_no_bc(jnod) 
           end do
!           stop 7
        end do
!        stop 229
! map to sfc ordering
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           resid_sfc_all(inod_sfc) = resid_no_bc(inod)
        end do
! 
        delta_psi_sfc = 0.0
        ncycle=nlevel-2
        ncycle=0
     do icycle=0,ncycle
        icycle_start=nlevel-icycle
        icycle_top=nlevel-1-icycle
        if(icycle==0) icycle_start=2
        if(.true.) then
           ilevel=icycle_start
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 resid_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc(jcol_sfc_all) 
              end do  
           end do    
        endif
! coarsen the residual...
        do ilevel=icycle_start,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         resid_sfc_all(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         resid_sfc_all(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           if(iscale_matrices==1) then ! assume the matricies have been divided through by mass matrix...
              ipt=fin_sfc_nonods(ilevel)
              resid_sfc_all(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
           endif
!           print *,'ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1):', &
!                    ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
        end do
!        stop 227
! 
! perform Jacobi relaxation on the 1d SFC and on each SFC grid level up to the 2nd or 1st. 
!        rhs_sfc_all = 0.0
!         print *,'ifilt_diag:',ifilt_diag
! 
! Jacobi relaxation on the all grids...
!        ilevel_finish=2-i_jacobi_on_finest_sfc ! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont
!        do ilevel=nlevel,max(icycle_top,2),-1
        do ilevel=nlevel,2,-1
!        do ilevel=nlevel,max(top_level,2),-1
!           print *,'ilevel=',ilevel
!        do ilevel=nlevel-1,1,-1
        if(ilevel.ge.top_level) then
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           end do
!           print *,'h1'
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc(jcol_sfc_all) 
              end do  
           end do    
!           print *,'h2'
! we have included the diagonal so take it away again...
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish    
              rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    +a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) * delta_psi_sfc(inod_sfc_all)
!              the_elthorne_variable   = 7e7*a_sfc(imid_ifilt,inod_sfc_all) * delta_psi_sfc(inod_sfc_all)
           end do
! Jacobi relaxation...
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              delta_psi_sfc(inod_sfc_all)  = &
                     relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) &
                                           +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
           end do
         endif
! map to finer grid
   if(ilevel.ne.1) then
           sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
           sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
           call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
   endif
! recalculate residual *******
    if(.false.) then
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 resid_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc(jcol_sfc_all) 
              end do  
           end do    
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              delta_psi_sfc(inod_sfc_all)=0.0
           end do
    endif
! recalculate residual *******
        end do ! do ilevel=nlevel,ilevel_finish,-1
     end do ! do icycle=1,ncycle
! 
! For ilevel=1 use Jacobi iteration with the original matrix...
        ilevel=1
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           delta_psi_temp(inod)=delta_psi_sfc(inod_sfc) 
           delta_psi(inod)=delta_psi_sfc(inod_sfc) 
        end do
   if(top_level==1) then
!   if(.true.) then
        delta_psi=delta_psi_temp
     if(i_jacobi_full_matrix==1) then ! include Jacobi relaxation on full matrix.
        do inod=1,nonods_no_bc
           rr=resid_no_bc(inod)
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              rr = rr - a_no_bc(count)*delta_psi_temp(jnod) 
           end do
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              if(jnod==inod) then ! diagonal...
                 rr = rr + a_no_bc(count)*delta_psi_temp(jnod) 
                 delta_psi(inod) = relax(ilevel) * rr /a_no_bc(count) &
                                   +(relax(ilevel)-1.0) * delta_psi_temp(inod)
              endif
           end do
        end do
      endif
   endif
        psi_no_bc = psi_no_bc + delta_psi
        print *,'just leaving sfc_solver_it_unstructured_fcycle'
!        stop 2921
        
        return 
        end subroutine sfc_solver_it_unstructured_fcycle
! 
! 
! 
! 
! Python interface is: 
! psi_no_bc = sfc_solver_it_unstructured_vcycle(psi_no_bc_guess, a_sfc, fin_sfc_nonods, &
!                                 nonods_sfc_all_grids, nlevel, &
!                                 a_no_bc, b_no_bc, relax,relax_down, &
!                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
!                                 ncola_no_bc, nonods_no_bc, &
!                                 nfilt_size_sfc)  
        subroutine sfc_solver_it_unstructured_vcycle(psi_no_bc, psi_no_bc_guess, a_sfc_all_un, &
                                 fina_sfc_all_un, cola_sfc_all_un, &
                                 ncola_sfc_all_un, fin_sfc_nonods, &
                                 nonods_sfc_all_grids, nlevel, &
                                 a_no_bc, b_no_bc, relax,relax_down, &
                                 fina_no_bc,cola_no_bc, sfc_node_ordering, &
                                 ncola_no_bc, nonods_no_bc, iscale_matrices, &
                                 i_jacobi_on_finest_sfc, i_jacobi_full_matrix)  
! Solve the system a_no_bc*psi_no_bc = b_no_bc with a single multi-grid iteration and 1 SFC. 
! it uses a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. 
! It does this with a kernal size of nfilt_size_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! if(iscale_matrices==1) then assume the matricies have been divided through by mass matrix.
! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont.
! i_jacobi_full_matrix=1 then include Jacobi relaxation on full matrix, =0 dont.
! Default values: iscale_matrices=0, i_jacobi_on_finest_sfc=0, i_jacobi_full_matrix=1
! one must have either i_jacobi_on_finest_sfc=1 and/or i_jacobi_full_matrix=1
! 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola_no_bc, nonods_no_bc, nonods_sfc_all_grids, nlevel
        integer, intent( in ) :: iscale_matrices, i_jacobi_on_finest_sfc, i_jacobi_full_matrix
        integer, intent( in ) :: ncola_sfc_all_un
        real, intent( out ) :: psi_no_bc(nonods_no_bc)
        real, intent( in ) :: a_sfc_all_un(ncola_sfc_all_un) 
        real, intent( in ) :: psi_no_bc_guess(nonods_no_bc), relax(nlevel), relax_down(nlevel) 
        real, intent( in ) :: a_no_bc(ncola_no_bc), b_no_bc(nonods_no_bc)
        integer, intent( in ) :: fin_sfc_nonods(nlevel+1)
        integer, intent( in ) :: fina_sfc_all_un(nonods_sfc_all_grids+1)
        integer, intent( in ) :: cola_sfc_all_un(ncola_sfc_all_un)
        integer, intent( in ) :: fina_no_bc(nonods_no_bc+1), cola_no_bc(ncola_no_bc)
        integer, intent( in ) :: sfc_node_ordering(nonods_no_bc)
! local variables...
        real, allocatable :: resid_no_bc(:), resid_sfc_all(:), rhs_sfc_all(:)
        real, allocatable :: delta_psi_temp(:),delta_psi(:),delta_psi_sfc(:),delta_psi_sfc2(:)
        integer, allocatable :: mida_sfc_all_un(:)
        real rr
        integer inod, count, jnod, inod_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer inod_sfc_all, ipt, inod_sfc_fine, inod_sfc_fine_all,inod_sfc_fine_all2
        integer ilevel_finish, inod_sfc_all_start, inod_sfc_all_finish
        integer jcol_sfc_all, jnod_sfc_all
! 
! calculate nlevel from nonods
        print *,'just inside sfc_solver_it_unstructured_vcycle'
!        stop 2922
        allocate(resid_no_bc(nonods_no_bc), resid_sfc_all(nonods_sfc_all_grids))
        allocate(delta_psi_sfc(nonods_sfc_all_grids), delta_psi(nonods_no_bc))
        allocate(delta_psi_sfc2(nonods_sfc_all_grids))
        allocate(rhs_sfc_all(nonods_sfc_all_grids), delta_psi_temp(nonods_no_bc))
        allocate(mida_sfc_all_un(nonods_sfc_all_grids))
! 
! find the diagonal...
        do inod_sfc_all = 1, nonods_sfc_all_grids  
           do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
              jnod_sfc_all=cola_sfc_all_un(count)
              if(jnod_sfc_all==inod_sfc_all) mida_sfc_all_un(inod_sfc_all)=count
           end do
        end do
! 
        psi_no_bc = psi_no_bc_guess ! need to do this because of python interface (can not have an inout variable)
        resid_no_bc=b_no_bc
        do inod=1,nonods_no_bc
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
!              print *,'inod,jnod,count,a_no_bc(count):',inod,jnod,count,a_no_bc(count)
              resid_no_bc(inod)=resid_no_bc(inod)-a_no_bc(count)*psi_no_bc(jnod) 
           end do
!           stop 7
        end do
!        stop 229
! map to sfc ordering
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           resid_sfc_all(inod_sfc) = resid_no_bc(inod)
        end do
! 
        delta_psi_sfc2=0.0
! coarsen the residual...
!        do ilevel=2,nlevel
        do ilevel=1,nlevel
      if(ilevel.ne.1) then
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         resid_sfc_all(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         resid_sfc_all(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           call map_sfc_fine_grid_2_course_grid_vec( &
!                                         delta_psi_sfc2(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
!                                         delta_psi_sfc2(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           if(iscale_matrices==1) then ! assume the matricies have been divided through by mass matrix...
              ipt=fin_sfc_nonods(ilevel)
              resid_sfc_all(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
!              delta_psi_sfc2(ipt:ipt+sfc_nonods_course-1)=0.5*resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
           endif
      endif
! 
      if(.true.) then ! perform a Jacobi relaxation and form a residual again resid_sfc_all. 
      if(ilevel.ne.nlevel) then
      if(ilevel.ne.1) then
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all)
              delta_psi_sfc2(inod_sfc_all) = relax_down(ilevel) * resid_sfc_all(inod_sfc_all) &
                                            /a_sfc_all_un(mida_sfc_all_un(inod_sfc_all))
           end do
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc2(jcol_sfc_all) 
              end do  
           end do  
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              resid_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) 
           end do
      endif
      endif
      endif
!           print *,'ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1):', &
!                    ilevel,resid_sfc_all(ipt:ipt+sfc_nonods_course-1)
        end do
!        stop 227
! 
! perform Jacobi relaxation on the 1d SFC and on each SFC grid level up to the 2nd or 1st. 
        delta_psi_sfc = 0.0
        rhs_sfc_all = 0.0
!         print *,'ifilt_diag:',ifilt_diag
! 
! Jacobi relaxation on the all grids...
        ilevel_finish=2-i_jacobi_on_finest_sfc ! i_jacobi_on_finest_sfc=1 include sfc relaxation on finnest grid, =0 dont
        do ilevel=nlevel,ilevel_finish,-1
!           print *,'ilevel=',ilevel
!        do ilevel=nlevel-1,1,-1
           inod_sfc_all_start  = fin_sfc_nonods(ilevel)
           inod_sfc_all_finish = fin_sfc_nonods(ilevel+1)-1
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              rhs_sfc_all(inod_sfc_all) = resid_sfc_all(inod_sfc_all) 
           end do
!           print *,'h1'
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish
              do count=fina_sfc_all_un(inod_sfc_all),fina_sfc_all_un(inod_sfc_all+1)-1
                 jcol_sfc_all = cola_sfc_all_un(count) 
                 rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    -a_sfc_all_un(count) * delta_psi_sfc(jcol_sfc_all) 
              end do  
           end do    
!           print *,'h2'
! we have included the diagonal so take it away again...
           do inod_sfc_all = inod_sfc_all_start, inod_sfc_all_finish    
              rhs_sfc_all(inod_sfc_all) = rhs_sfc_all(inod_sfc_all) &
                                    +a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) * delta_psi_sfc(inod_sfc_all)
!              the_elthorne_variable   = 7e7*a_sfc(imid_ifilt,inod_sfc_all) * delta_psi_sfc(inod_sfc_all)
           end do
! Jacobi relaxation...
           do inod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1 
              delta_psi_sfc(inod_sfc_all)  = &
                     relax(ilevel) * rhs_sfc_all(inod_sfc_all) / a_sfc_all_un(mida_sfc_all_un(inod_sfc_all)) &
                                           +(1.-relax(ilevel))* delta_psi_sfc(inod_sfc_all)
           end do
! map to finer grid
   if(ilevel.ne.1) then
           sfc_nonods_course = fin_sfc_nonods(ilevel+1) - fin_sfc_nonods(ilevel)
           sfc_nonods_fine   = fin_sfc_nonods(ilevel)   - fin_sfc_nonods(ilevel-1)
           call map_sfc_course_grid_2_fine_grid_vec( &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                                         delta_psi_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course)
      if(.false.) then ! reduce delta on fine grid by factor of 2.
         ipt=fin_sfc_nonods(ilevel-1)
         delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)=0.5*delta_psi_sfc(ipt:ipt+sfc_nonods_fine-1)
      endif
   endif
      if(.false.) then
         delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel+1)-1) &
        =delta_psi_sfc(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel+1)-1) &
        +delta_psi_sfc2(fin_sfc_nonods(ilevel):fin_sfc_nonods(ilevel+1)-1) 
      endif
        end do ! do ilevel=nlevel,ilevel_finish,-1
! 
! For ilevel=1 use Jacobi iteration with the original matrix...
        ilevel=1
        do inod=1,nonods_no_bc
           inod_sfc=sfc_node_ordering(inod)
           delta_psi_temp(inod)=delta_psi_sfc(inod_sfc) 
        end do
        delta_psi=delta_psi_temp
     if(i_jacobi_full_matrix==1) then ! include Jacobi relaxation on full matrix.
        do inod=1,nonods_no_bc
           rr=resid_no_bc(inod)
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              rr = rr - a_no_bc(count)*delta_psi_temp(jnod) 
           end do
           do count=fina_no_bc(inod),fina_no_bc(inod+1)-1
              jnod=cola_no_bc(count)
              if(jnod==inod) then ! diagonal...
                 rr = rr + a_no_bc(count)*delta_psi_temp(jnod) 
                 delta_psi(inod) = relax(ilevel) * rr /a_no_bc(count) &
                                   +(relax(ilevel)-1.0) * delta_psi_temp(inod)
              endif
           end do
        end do
      endif
        psi_no_bc = psi_no_bc + delta_psi
        print *,'just leaving sfc_solver_it_unstructured'
!        stop 2921
        
        return 
        end subroutine sfc_solver_it_unstructured_vcycle
! 
! 
! 
! 
! in python: 
! t_old, t_new, x_all, ndglno, fina, cola  = set_up_grid_problem(dx, dy, ncola, nx, ny, nloc, ndim)
        subroutine set_up_grid_problem(t_old, t_new, x_all, ndglno, fina, cola, &
                                       dx, dy, ncola, nx, ny, nloc, ndim) 
        implicit none
! set up grid
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        
! local variables...
! 2d:
! nx=no of nodes across, ny=no of nodes up. 
        integer, intent( in ) :: ncola, nx, ny, nloc, ndim
!        integer, parameter :: nonods=nx*ny, totele=(nx-1)*(ny-1)
        real, intent( out ) :: t_old(nx*ny), t_new(nx*ny), x_all(ndim,(nx+1)*(ny+1)*nloc) 
        integer, intent( out ) ::  ndglno(nloc*(nx+1)*(ny+1)), fina(nx*ny+1), cola(ncola) 
        real, intent( in ) :: dx, dy
! example of node ordering...
!      7-----8-----9
!      !     !     !
!      !     !     !        
!      4-----5-----6
!      !     !     !
!      !     !     !        
!      1-----2-----3
! filters for ann:
! local variables...
        integer npoly,ele_type, iloc,count
        integer enx_i, eny_j, ele, i,j, ii,jj, iii,jjj, inod, jnod, itime
! ! 
        print *,'here in setup grid'
!        stop 23
! 

        t_old=0.0
        do enx_i=1,nx-1
        do eny_j=1,ny-1
           ele=(eny_j-1)*(nx-1)+enx_i
!
           do ii=1,2
           do jj=1,2

              iloc=(jj-1)*2+ii 
              inod = (eny_j-2 +jj)*nx + enx_i + ii-1
              ndglno((ele-1)*nloc+iloc)=inod 
              x_all(1,(ele-1)*nloc+iloc) = real(enx_i-2+ii ) * dx
              x_all(2,(ele-1)*nloc+iloc) = real(eny_j-2+jj ) * dy 
              if(  (x_all(1,(ele-1)*nloc+iloc)>1.9) &
              .and.(x_all(1,(ele-1)*nloc+iloc)<3.9) ) then
                 if(  (x_all(2,(ele-1)*nloc+iloc)>1.9) &
                 .and.(x_all(2,(ele-1)*nloc+iloc)<3.9) ) then
                     t_old(inod)=1.0
                 endif
              endif
           end do
           end do
        end do
        end do
        t_new=t_old
!        print *,'t_new:',t_new
!        stop 22
!
        count=0
        do j=1,ny
        do i=1,nx
           inod=(j-1)*nx+i
           fina(inod)=count+1
           do jj=-1,1
           do ii=-1,1
              iii=i+ii
              jjj=j+jj
              if((iii>0).and.(jjj>0)) then 
              if((iii<nx+1).and.(jjj<ny+1)) then 
                 jnod=(jjj-1)*nx + iii
                 count=count+1
                 cola(count)=jnod
              endif
              endif
           end do
           end do
        end do
        end do
        fina(nx*ny+1)=count+1 
        end subroutine set_up_grid_problem
! 
! 
! 
! 
! in python:
! nlevel = calculate_nlevel_sfc(nonods)
        subroutine calculate_nlevel_sfc(nlevel,nonods)
! this subroutine calculates the number of multi-grid levels for a 
! space filling curve multi-grid or 1d multi-grid applied to nonods nodes. 
        implicit none
        integer, intent( in ) :: nonods
        integer, intent( out ) :: nlevel
! local variables...
        integer sfc_nonods_fine,sfc_nonods_course,ilevel
! coarsen...
        if(nonods==1) then
           nlevel=1 
        else
           sfc_nonods_course=nonods
           do ilevel=2,200
              sfc_nonods_fine=sfc_nonods_course
              sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
              nlevel=ilevel
              if(sfc_nonods_course==1) exit
           end do
        endif
        return
        end subroutine calculate_nlevel_sfc
! 
! 
! 
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = best_sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
      subroutine best_sfc_mapping_to_sfc_matrix_3(a_sfc, b_sfc, ml_sfc, &
                                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml, relax_keep_off, &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, max_nlevel, &
                                     nfilt_size_sfc) 
! It does this with a kernal size of 3. 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It does this with a kernal size of 3. 
! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods). 
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids, max_nlevel, nfilt_size_sfc
        real, intent( out ) :: a_sfc(nfilt_size_sfc,max_nonods_sfc_all_grids), &
                               b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        real, intent( in ) :: relax_keep_off
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel+1), nlevel
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:)
        real, allocatable :: pot_diag_a_sfc(:)
        integer i, count, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod, ipt
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
! 
        print *,'2-just inside best_sfc_mapping_to_sfc_matrix_3'
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! form SFC matrix...
        a_sfc(:,:)=0.0
        b_sfc(:)=0.0
        ml_sfc(:)=0.0
       do ifinest_nod=1,nonods
          ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
          b_sfc(ifinest_nod_sfc)=b(ifinest_nod)
          ml_sfc(ifinest_nod_sfc)=ml(ifinest_nod)
       end do 
!       print *,'here 1 nlevel:',nlevel
! 
! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
!           print *,'ilevel=',ilevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
!           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
!                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1):', &
!                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1)
!           print *,'max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine:', &
!                    max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine

           call map_sfc_fine_grid_2_course_grid_vec( &
                                         ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           ipt=fin_sfc_nonods(ilevel)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_course-1):',ml_sfc(ipt:ipt+sfc_nonods_course-1)
!           ipt=fin_sfc_nonods(ilevel-1)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_fine-1):',ml_sfc(ipt:ipt+sfc_nonods_fine-1)
!       print *,'here 1.1' 
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!       print *,'here 1.2' 
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum-1
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
       print *,'here 2'
!         stop 25
! sfc_node_ordering(nod) = new node numbering from current node number nod.
        allocate(sfc_node_ordering_inverse(nonods))
        do ifinest_nod=1,nonods
           ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
           sfc_node_ordering_inverse(ifinest_nod_sfc) = ifinest_nod
        end do 
!       print *,'here 3'
! 
! coarsen the original matrix to form a_sfc on each grid level...
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
        allocate(pot_diag_a_sfc(nonods_sfc_all_grids))
        pot_diag_a_sfc=0.0
        do ilevel=1,nlevel
!           print *,'ilevel=',ilevel
           ilevel2=2**(ilevel-1)
           do ifinest_nod_sfc=1,nonods
              icourse_nod_sfc = 1 + (ifinest_nod_sfc-1)/ilevel2
              icourse_nod_sfc_displaced = icourse_nod_sfc + fin_sfc_nonods(ilevel) - 1
              ifinest_nod = sfc_node_ordering_inverse(ifinest_nod_sfc) 

              do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                 jfinest_nod=cola(count)
                 jfinest_nod_sfc = sfc_node_ordering(jfinest_nod)
                 jcourse_nod_sfc = 1 + (jfinest_nod_sfc-1)/ilevel2
                 if(jcourse_nod_sfc==icourse_nod_sfc-1) then
                    a_sfc(1,icourse_nod_sfc_displaced)=a_sfc(1,icourse_nod_sfc_displaced)+a(count)
                 else if(jcourse_nod_sfc==icourse_nod_sfc+1) then
                    a_sfc(3,icourse_nod_sfc_displaced)=a_sfc(3,icourse_nod_sfc_displaced)+a(count)
                 else if(jcourse_nod_sfc==icourse_nod_sfc) then
                    a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced)+a(count) ! diagonal
                 else
!                    if(ilevel>nlevel-7) then
!                    a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced)+a(count)
!                    else
                    a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced)+relax_keep_off*a(count) ! diagonal
!                    endif
!                    pot_diag_a_sfc(icourse_nod_sfc_displaced)=  &
!                         pot_diag_a_sfc(icourse_nod_sfc_displaced) + a(count) ! diagonal
                 endif
              end do

           end do
        end do

      if(.false.) then
        do icourse_nod_sfc_displaced=1,nonods_sfc_all_grids
          if(.true.) then ! simplest and best
             if(icourse_nod_sfc_displaced.lt.nonods_sfc_all_grids-70) then
              a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced) &
               +relax_keep_off*pot_diag_a_sfc(icourse_nod_sfc_displaced) ! diagonal
             endif
          else ! variable method...
           if(abs(a_sfc(1,icourse_nod_sfc_displaced))+abs(a_sfc(3,icourse_nod_sfc_displaced)).gt.1.e-2) then
!           if(a_sfc(2,icourse_nod_sfc_displaced)+pot_diag_a_sfc(icourse_nod_sfc_displaced).gt.1.e-2) then
              a_sfc(2,icourse_nod_sfc_displaced)=a_sfc(2,icourse_nod_sfc_displaced) &
              +1.0*pot_diag_a_sfc(icourse_nod_sfc_displaced) ! diagonal
!              +relax_keep_off*pot_diag_a_sfc(icourse_nod_sfc_displaced) ! diagonal
           endif
          endif
!        do ifinest_nod_sfc=80,84
           if(a_sfc(2,icourse_nod_sfc_displaced).lt.1.e-4) then
              print *,'icourse_nod_sfc_displaced,a_sfc(:,icourse_nod_sfc_displaced):', &
                       icourse_nod_sfc_displaced,a_sfc(:,icourse_nod_sfc_displaced)
              stop 2211
           end if
        end do
      endif
!        stop 282
        print *,'just leaving best_sfc_mapping_to_sfc_matrix_3'
        return 
        end subroutine best_sfc_mapping_to_sfc_matrix_3
! 
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = best_sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
      subroutine best_sfc_mapping_to_sfc_matrix_n(a_sfc, b_sfc, ml_sfc, &
                                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml, relax_keep_off, &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, max_nlevel, &
                                     nfilt_size_sfc)  
! It does this with a kernal size of nfilt_size_sfc. 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It does this with a kernal size of nfilt_size_sfc. 
! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods). 
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids, max_nlevel, nfilt_size_sfc
        real, intent( out ) :: a_sfc(nfilt_size_sfc,max_nonods_sfc_all_grids), &
                               b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        real, intent( in ) :: relax_keep_off
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel+1), nlevel
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:)
        real, allocatable :: pot_diag_a_sfc(:)
        integer i, count, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_displaced
        integer ifinest_nod, jfinest_nod, ipt
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer ifilt, icent_ifilt, ifilt_diag
        logical found
! 
        print *,'2-just inside best_sfc_mapping_to_sfc_matrix_n'
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! 
        ifilt_diag = 1 + nfilt_size_sfc/2
! form SFC matrix...
        a_sfc(:,:)=0.0
        b_sfc(:)=0.0
        ml_sfc(:)=0.0
       do ifinest_nod=1,nonods
          ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
          b_sfc(ifinest_nod_sfc)=b(ifinest_nod)
          ml_sfc(ifinest_nod_sfc)=ml(ifinest_nod)
       end do 
!       print *,'here 1 nlevel:',nlevel
! 
! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
!           print *,'ilevel=',ilevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
!           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
!                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1):', &
!                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1)
!           print *,'max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine:', &
!                    max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine

           call map_sfc_fine_grid_2_course_grid_vec( &
                                         ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           ipt=fin_sfc_nonods(ilevel)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_course-1):',ml_sfc(ipt:ipt+sfc_nonods_course-1)
!           ipt=fin_sfc_nonods(ilevel-1)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_fine-1):',ml_sfc(ipt:ipt+sfc_nonods_fine-1)
!       print *,'here 1.1' 
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!       print *,'here 1.2' 
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum-1
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
       print *,'here 2'
!         stop 25
! sfc_node_ordering(nod) = new node numbering from current node number nod.
        allocate(sfc_node_ordering_inverse(nonods))
        do ifinest_nod=1,nonods
           ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
           sfc_node_ordering_inverse(ifinest_nod_sfc) = ifinest_nod
        end do 
!       print *,'here 3'
! 
! coarsen the original matrix to form a_sfc on each grid level...
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
        allocate(pot_diag_a_sfc(nonods_sfc_all_grids))
        pot_diag_a_sfc=0.0
        do ilevel=1,nlevel
!           print *,'ilevel=',ilevel
           ilevel2=2**(ilevel-1)
           do ifinest_nod_sfc=1,nonods
              icourse_nod_sfc = 1 + (ifinest_nod_sfc-1)/ilevel2
              icourse_nod_sfc_displaced = icourse_nod_sfc + fin_sfc_nonods(ilevel) - 1
              ifinest_nod = sfc_node_ordering_inverse(ifinest_nod_sfc) 

              do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                 jfinest_nod=cola(count)
                 jfinest_nod_sfc = sfc_node_ordering(jfinest_nod)
                 jcourse_nod_sfc = 1 + (jfinest_nod_sfc-1)/ilevel2
                 found=.false. 
                 do ifilt = 1, nfilt_size_sfc
                     icent_ifilt = ifilt - ifilt_diag
                    if(jcourse_nod_sfc==icourse_nod_sfc+icent_ifilt) then ! off diagonal and diagonal..
                       a_sfc(ifilt,icourse_nod_sfc_displaced)=a_sfc(ifilt,icourse_nod_sfc_displaced)+a(count)
                       found=.true. 
                    endif
                 end do
                 if(.not.found) then ! could not find it in the matrix sparcity so put on diagonal.
                   a_sfc(ifilt_diag,icourse_nod_sfc_displaced) &
                       =a_sfc(ifilt_diag,icourse_nod_sfc_displaced)+relax_keep_off*a(count) ! diagonal
                 endif
              end do ! do count=fina(ifinest_nod),fina(ifinest_nod+1)-1

           end do
        end do

!        stop 282
        print *,'just leaving best_sfc_mapping_to_sfc_matrix_n'
        return 
        end subroutine best_sfc_mapping_to_sfc_matrix_n
! 
! 
! 
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = best_sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
      subroutine best_sfc_mapping_to_sfc_matrix_unstructured( &
                                     a_sfc_all_un,fina_sfc_all_un, &
                                     cola_sfc_all_un,ncola_sfc_all_un, &
                                     b_sfc, ml_sfc, &
                                     fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml,  &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, &
                                     max_ncola_sfc_all_un, max_nlevel)  
! It does this with a kernal size of nfilt_size_sfc. 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It does this with a kernal size of nfilt_size_sfc. 
! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods). 
!        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
!        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
!        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids
        integer, intent( in ) :: max_ncola_sfc_all_un, max_nlevel
        real, intent( out ) :: a_sfc_all_un(max_ncola_sfc_all_un), &
                               b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel+1), nlevel
        integer, intent( out ) :: fina_sfc_all_un(max_nonods_sfc_all_grids+1), &
                                  cola_sfc_all_un(max_ncola_sfc_all_un),ncola_sfc_all_un
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:), in_row_count(:)
        integer, allocatable :: fina_sfc_all_un2(:)
        integer i, count, count2, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_all
        integer ifinest_nod, jfinest_nod, ipt
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer nrow, count_col, idisplace, jcourse_nod_sfc_all2
        integer jcourse_nod_sfc_all, ifinest_nod_sfc_all, count_all
        logical found
! 
        print *,'2-just inside best_sfc_mapping_to_sfc_matrix_unstructured'
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! 
! form SFC matrix...
!        a_sfc(:,:)=0.0
        b_sfc(:)=0.0
        ml_sfc(:)=0.0
        do ifinest_nod=1,nonods
           ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
           b_sfc(ifinest_nod_sfc)=b(ifinest_nod)
           ml_sfc(ifinest_nod_sfc)=ml(ifinest_nod)
        end do 
!       print *,'here 1 nlevel:',nlevel
! 
! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
!           print *,'ilevel=',ilevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
!           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
!                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1):', &
!                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1)
!           print *,'max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine:', &
!                    max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine

           call map_sfc_fine_grid_2_course_grid_vec( &
                                         ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!           ipt=fin_sfc_nonods(ilevel)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_course-1):',ml_sfc(ipt:ipt+sfc_nonods_course-1)
!           ipt=fin_sfc_nonods(ilevel-1)
!           print *,'ml_sfc(ipt:ipt+sfc_nonods_fine-1):',ml_sfc(ipt:ipt+sfc_nonods_fine-1)
!       print *,'here 1.1' 
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
!       print *,'here 1.2' 
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum-1
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
        print *,'here 2'
!         stop 25
! sfc_node_ordering(nod) = new node numbering from current node number nod.
        allocate(sfc_node_ordering_inverse(nonods))
        do ifinest_nod=1,nonods
           ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
           sfc_node_ordering_inverse(ifinest_nod_sfc) = ifinest_nod
        end do 
!     
        print *,'---max_ncola_sfc_all_un,ncola,nlevel:',max_ncola_sfc_all_un,ncola,nlevel
        print *,'---nonods,nonods_sfc_all_grids:',nonods,nonods_sfc_all_grids
        print *,'fin_sfc_nonods(1:,nlevel+1):',fin_sfc_nonods(1:nlevel+1)
!        a_sfc_all_un=0.0
        count_all=0
        do ilevel=1,nlevel
           ilevel2=2**(ilevel-1)
           print *,'--- ilevel=',ilevel
           idisplace = fin_sfc_nonods(ilevel) 
           fina_sfc_all_un(idisplace) = count_all+1
           do ifinest_nod_sfc=1,nonods
!              print *,'ifinest_nod_sfc=',ifinest_nod_sfc
              ifinest_nod = sfc_node_ordering_inverse(ifinest_nod_sfc)
              icourse_nod_sfc_all = idisplace + (ifinest_nod_sfc-1)/ilevel2
              do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                 jfinest_nod = cola(count)
                 jfinest_nod_sfc = sfc_node_ordering(jfinest_nod) 
                 jcourse_nod_sfc_all = idisplace + (jfinest_nod_sfc-1)/ilevel2
! look to see if we have included jcourse_nod_sfc_all yet
                 found=.false.
                 do count2=fina_sfc_all_un(icourse_nod_sfc_all),count_all
                    jcourse_nod_sfc_all2=cola_sfc_all_un(count2)
                    if(jcourse_nod_sfc_all==jcourse_nod_sfc_all2) then
                       found=.true.
                       a_sfc_all_un(count2)=a_sfc_all_un(count2)+a(count) ! map from original matrix
                    endif
                 end do
                 if(.not.found) then
                    count_all=count_all+1
                    cola_sfc_all_un(count_all) = jcourse_nod_sfc_all
!                    a_sfc_all_un(count_all)=a_sfc_all_un(count_all)+a(count) ! map from original matrix
                    a_sfc_all_un(count_all)=a(count) ! map from original matrix
                 endif
              end do ! do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
              fina_sfc_all_un(icourse_nod_sfc_all+1) = count_all+1
           end do ! do ifinest_nod_sfc=1,nonods
           print *,'here2'
           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1):', &
                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)
           print *,'diff=',fin_sfc_nonods(ilevel+1)-fin_sfc_nonods(ilevel)
           print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel)):',fina_sfc_all_un(fin_sfc_nonods(ilevel))
           print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1:',fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1
           print *,'difference:', &
               fina_sfc_all_un(fin_sfc_nonods(ilevel+1)) - fina_sfc_all_un(fin_sfc_nonods(ilevel))
          if(.false.) then
           do icourse_nod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1
              print *,'icourse_nod_sfc_all=',icourse_nod_sfc_all
              print *,'cola_sfc_all_un(count2):', &
                  (cola_sfc_all_un(count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                                                  fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
              print *,'a_sfc_all_un(count2):', &
                  (a_sfc_all_un(count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                                               fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
           end do
          endif
        end do ! do ilevel=1,nlevel

        ncola_sfc_all_un = fina_sfc_all_un(nonods_sfc_all_grids+1)-1
        if( max_ncola_sfc_all_un < ncola_sfc_all_un ) then
           print *,'run out of memory here stopping'
           stop 2825
        endif
! 
        print *,'ncola, ncola_sfc_all_un:',ncola, ncola_sfc_all_un
        print *,'nonods_sfc_all_grids:',nonods_sfc_all_grids
        print *,'fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids):', &
                 fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids)

!        stop 282
        print *,'just leaving best_sfc_mapping_to_sfc_matrix_unstructured'
        return 
        end subroutine best_sfc_mapping_to_sfc_matrix_unstructured
! 
! 
! 
! python interface: 
! filt_nxnx, filt_nnx, ml = filters_for_structured_mesh_dg(dx,ndim,nloc)
        subroutine filters_for_structured_mesh_dg_de_bug(filt_nxnx, filt_nnx, ml, dx,ndim,nloc)
        implicit none
! this subroutine calculates filters for the 2D rectangular element and 
! the 3D hex element of dimensions dx.
! filt_nxnx is the filter for the diffusion/Laplacian operator. 
! filt_nnx is the filter for the derivatives in the x,y and z-directions. 
! ml contains the mass associated with the local nodes. 
! nloc=no of nodes per element
! ndim=no of dimensions 
! dx is the dimensions of the element - width, length, height. 
! For the arrays:
! filt_nxnx(3,3,1+(ndim-2)*2, nloc)
! filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! The 3,3 is the dimensions of the filter in 2D. It needs to be a 3x3 array.
! NDIM is the number of dimensions ndim=2 in our case but will change to3 soon.
! nloc is the number of local noes in an element. So we produce a filter for each local node of an element - 4 in our case.
! nsign=1 or 2 and is 1 is we have a positive sign of the component of interest (idim that goes into
! the ndim part of the array) of the velocity. Its 2 if the component is negative. 
        integer, parameter :: nsign=2
        integer, intent( in ) :: ndim, nloc
        real, intent( in ) :: dx(ndim)
        real, intent( out ) :: filt_nxnx(3,3,1+(ndim-2)*2, ndim,nloc)
        real, intent( out ) :: filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
        real, intent( out ) :: ml(nloc)
        
! local variables...
        real, allocatable :: nxnx_ele(:,:,:),nxn_ele(:,:,:,:),snsn(:,:,:)
        real, allocatable :: filt_nnx_nosign(:,:,:, :,:)
        real, allocatable :: dx_face_normal(:,:),face_normal(:,:,:) 
        real, allocatable :: x_all(:,:)
        integer totele,ele, iloc,isig
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k
        integer snloc,nface,idim,iface
! 
        print *,'just inside filters_for_structured_mesh_dg_de_bug'
!        return
        totele=1
        if(ndim==2) then
! 2d:
           snloc=2
           nface=4
        else ! ndim=3
! 3d:
           snloc=4
           nface=6
        endif
        allocate(nxnx_ele(nloc,nloc,totele),nxn_ele(ndim,nloc,nloc,totele))
        allocate(filt_nnx_nosign(3,3,1+(ndim-2)*2, ndim,nloc))
        allocate(snsn(nloc,nloc,totele))
        allocate(dx_face_normal(nface,totele),face_normal(ndim,nface,totele))
        allocate(x_all(ndim,totele*nloc)) 
! 
        if(ndim==2) then
          x_all(:,1)=(/-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0/)
          x_all(:,4)=(/+1.0,+1.0/)
           print *,'--x_all:',x_all
           print *,'dx:',dx
        endif
        if(ndim==3) then
          x_all(:,1)=(/-1.0,-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0,-1.0/)
          x_all(:,4)=(/+1.0,+1.0,-1.0/)
! 2nd layer...
          x_all(:,5)=(/-1.0,-1.0,+1.0/)
          x_all(:,6)=(/+1.0,-1.0,+1.0/)
          x_all(:,7)=(/-1.0,+1.0,+1.0/)
          x_all(:,8)=(/+1.0,+1.0,+1.0/)
        endif
        print *,'ndim,nloc:',ndim,nloc
! make the elements of width dx
        do iloc=1,nloc
           x_all(:,iloc)=x_all(:,iloc)*0.5*dx(:) 
        end do
! 
! form the sourface and volume integrals...
!        stop 298
        call spacial_tables_for_dg_filters(nxnx_ele, nxn_ele, ml, &
                                           dx_face_normal, face_normal, snsn, &
                                           x_all, totele, nface,ndim,nloc,snloc) 
                           
        filt_nxnx=1. 
        filt_nnx=1.
        ml=1.
        print *,'just leaving filters_for_structured_mesh_dg_de_bug'
        return               
! 
!        stop 299
! overwrite dx as its in the orthogonal direction (wrong direction)...
        ele=1    
        do iface=1,nface
           idim= 1 + (iface-1)/2  
           dx_face_normal(iface,ele) = dx(idim) 
        end do
! 
!        stop 2910
! form the spatial dg filters from the spatial tables...
        call form_dg_filters_from_nx(filt_nxnx, filt_nnx_nosign, &
             filt_nnx,  &
             nxnx_ele(:,:,ele),nxn_ele(:,:,:,ele),snsn(:,:,ele), &
             dx_face_normal(:,ele),face_normal(:,:,ele), &
             ndim,nface,nloc,snloc )
!        stop 2911
        print *,'just outside of form_dg_filters_from_nx'
! 
        do iloc=1,nloc
           do isig=1,2
              filt_nnx(:,:,:,:,isig,iloc) = filt_nnx(:,:,:,:,isig,iloc) &
                                          + filt_nnx_nosign(:,:,:,:,iloc)
           end do
        end do
        print *,'filt_nxnx:',filt_nxnx
        print *,'filt_nnx:',filt_nnx
        print *,'ml:',ml
        print *,'just about to leave filters_for_structured_mesh_dg'

        filt_nxnx=1. 
        filt_nnx=1.
        ml=1.
! 
        print *,'just leaving filters_for_structured_mesh_dg_de_bug'
        return
        end subroutine filters_for_structured_mesh_dg_de_bug
! 
! 
! python interface: 
! filt_nxnx, filt_nnx, ml = filters_for_structured_mesh_dg(dx,ndim,nloc)
        subroutine filters_for_structured_mesh_dg(filt_nxnx, filt_nnx, ml, dx,ndim,nloc)
        implicit none
! this subroutine calculates filters for the 2D rectangular element and 
! the 3D hex element of dimensions dx.
! filt_nxnx is the filter for the diffusion/Laplacian operator. 
! filt_nnx is the filter for the derivatives in the x,y and z-directions. 
! ml contains the mass associated with the local nodes. 
! nloc=no of nodes per element
! ndim=no of dimensions 
! dx is the dimensions of the element - width, length, height. 
! For the arrays:
! filt_nxnx(3,3,1+(ndim-2)*2, ndim,nloc)
! filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! The 3,3 is the dimensions of the filter in 2D. It needs to be a 3x3 array.
! NDIM is the number of dimensions ndim=2 in our case but will change to3 soon.
! nloc is the number of local noes in an element. So we produce a filter for each local node of an element - 4 in our case.
! nsign=1 or 2 and is 1 is we have a positive sign of the component of interest (idim that goes into
! the ndim part of the array) of the velocity. Its 2 if the component is negative. 
        integer, parameter :: nsign=2
        integer, intent( in ) :: ndim, nloc
        real, intent( in ) :: dx(ndim)
        real, intent( out ) :: filt_nxnx(3,3,1+(ndim-2)*2, ndim,nloc)
        real, intent( out ) :: filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
        real, intent( out ) :: ml(nloc)
        
! local variables...
        real, allocatable :: nxnx_ele(:,:,:,:),nxn_ele(:,:,:,:),snsn(:,:,:,:)
        real, allocatable :: filt_nnx_nosign(:,:,:, :,:)
        real, allocatable :: dx_face_normal(:,:),face_normal(:,:,:) 
        real, allocatable :: x_all(:,:)
        integer totele,ele, iloc,isig
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k
        integer snloc,nface,idim,iface
! 
        print *,'just inside filters_for_structured_mesh_dg'
!        return
        totele=1
        if(ndim==2) then
! 2d:
           snloc=2
           nface=4
        else ! ndim=3
! 3d:
           snloc=4
           nface=6
        endif
        allocate(nxnx_ele(ndim,nloc,nloc,totele),nxn_ele(ndim,nloc,nloc,totele))
        allocate(filt_nnx_nosign(3,3,1+(ndim-2)*2, ndim,nloc))
        allocate(snsn(snloc,snloc,nface,totele))
        allocate(dx_face_normal(nface,totele),face_normal(ndim,nface,totele))
        allocate(x_all(ndim,totele*nloc)) 
! 
        if(ndim==2) then
          x_all(:,1)=(/-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0/)
          x_all(:,4)=(/+1.0,+1.0/)
           print *,'--x_all:',x_all
           print *,'dx:',dx
        endif
        if(ndim==3) then
          x_all(:,1)=(/-1.0,-1.0,-1.0/)
          x_all(:,2)=(/+1.0,-1.0,-1.0/)
          x_all(:,3)=(/-1.0,+1.0,-1.0/)
          x_all(:,4)=(/+1.0,+1.0,-1.0/)
! 2nd layer...
          x_all(:,5)=(/-1.0,-1.0,+1.0/)
          x_all(:,6)=(/+1.0,-1.0,+1.0/)
          x_all(:,7)=(/-1.0,+1.0,+1.0/)
          x_all(:,8)=(/+1.0,+1.0,+1.0/)
        endif
        print *,'ndim,nloc:',ndim,nloc
! make the elements of width dx
        do iloc=1,nloc
           x_all(:,iloc)=x_all(:,iloc)*0.5*dx(:) 
        end do
         print *,'x_all:',x_all
        do idim=1,ndim
           print *,'idim=',idim
           print *,'x_all(idim,:):',x_all(idim,:)
        end do
!        stop 292
! 
! form the sourface and volume integrals...
!        stop 298
        call spacial_tables_for_dg_filters(nxnx_ele, nxn_ele, ml, &
                                           dx_face_normal, face_normal, snsn, &
                                           x_all, totele, nface,ndim,nloc,snloc)
!        stop 299
! overwrite dx as its in the orthogonal direction (wrong direction)...
        ele=1    
        do iface=1,nface
           idim= 1 + (iface-1)/2  
           dx_face_normal(iface,ele) = dx(idim)  
        end do
! 
!        stop 2910
! form the spatial dg filters from the spatial tables...
        call form_dg_filters_from_nx(filt_nxnx, filt_nnx_nosign, &
             filt_nnx,  &
             nxnx_ele(:,:,:,ele),nxn_ele(:,:,:,ele),snsn(:,:,:,ele), &
             dx_face_normal(:,ele),face_normal(:,:,ele), &
             ndim,nface,nloc,snloc )
!        stop 2911
        print *,'just outside of form_dg_filters_from_nx'
! 
        do iloc=1,nloc
           do isig=1,2
              filt_nnx(:,:,:,:,isig,iloc) = filt_nnx(:,:,:,:,isig,iloc) &
                                          + filt_nnx_nosign(:,:,:,:,iloc)
           end do
        end do
        print *,'filt_nxnx:',filt_nxnx
        print *,'filt_nnx:',filt_nnx
        print *,'ml:',ml
        print *,'just about to leave filters_for_structured_mesh_dg'
!         stop 2921
! 
        return
        end subroutine filters_for_structured_mesh_dg
! 
! 
! 
! 
! python interface:
! filt_nxnx,filt_nnx, filt_nnx_nosign, suf_filt_nnx, ml, face_normal = filters_for_unstructured_mesh_dg( &
!                                                     x_all,totele,nface,ndim,nloc)
        subroutine filters_for_unstructured_mesh_dg(filt_nxnx,filt_nnx, filt_nnx_nosign, suf_filt_nnx, ml, face_normal, &
                                                    x_all,totele,nface,ndim,nloc)
        implicit none
! This subroutine calculates filters for the 2D rectangular elements 
! and 3D hex elemenents that can be distorted. 
! filt_nxnx is the filter for the diffusion/Laplacian operator. 
! filt_nnx is the filter for the derivatives in the x,y and z-directions. 
! ml contains the mass associated with the local nodes. 
! nloc=no of nodes per element
! ndim=no of dimensions 
! For the arrays:
! filt_nxnx(3,3,1+(ndim-2)*2, ndim, nloc)
! filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! The 3,3 is the dimensions of the filter in 2D. It needs to be a 3x3 array.
! NDIM is the number of dimensions ndim=2 in our case but will change to3 soon.
! nloc is the number of local noes in an element. So we produce a filter for each local node of an element - 4 in our case.
! nsign=1 or 2 and is 1 is we have a positive sign of the component of interest (idim that goes into
! the ndim part of the array) of the velocity. Its 2 if the component is negative. 
        integer, parameter :: nsign=2
        integer, intent( in ) :: totele,nface,ndim,nloc
        real, intent( in ) :: x_all(ndim,totele)
        real, intent( out ) :: filt_nxnx(3,3,1+(ndim-2)*2, ndim,nloc,totele)
        real, intent( out ) :: filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc,totele)
        real, intent( out ) :: filt_nnx_nosign(3,3,1+(ndim-2)*2, ndim,nloc,totele)
        real, intent( out ) :: suf_filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc,totele)
        real, intent( out ) :: ml(totele*nloc)
        real, intent( out ) :: face_normal(ndim,nface,totele)
! 
! local variables...
        real, allocatable :: nxnx_ele(:,:,:),nxn_ele(:,:,:,:)
        real, allocatable :: snsn(:,:,:,:)
        real, allocatable :: dx_face_normal(:,:)
        integer snloc,ele
!
        allocate(nxnx_ele(nloc,nloc,totele),nxn_ele(ndim,nloc,nloc,totele))
        allocate(snsn(snloc,snloc,nface,totele))
        allocate(dx_face_normal(nface,totele))
!
        if(ndim==2) then ! 2D...
           snloc=2
        else ! 3D...
           snloc=4
        endif
! 
! Form the spatial tables that will be used to form dg filters...
        call spacial_tables_for_dg_filters(nxnx_ele, nxn_ele, ml, &
                                           dx_face_normal, face_normal, snsn,  &
                                           x_all, totele, nface,ndim,nloc,snloc)
! 
! Form the spatial dg filters from the spatial tables...
        do ele=1,totele
           call form_dg_filters_from_nx(filt_nxnx(:,:,:,:,:,ele), filt_nnx_nosign(:,:,:,:,:,ele), &
                filt_nnx(:,:,:,:,:,:,ele),  &
                nxnx_ele(:,:,ele),nxn_ele(:,:,:,ele),snsn(:,:,:,ele), &
                dx_face_normal(:,ele),face_normal(:,:,ele), &
                ndim,nface,nloc,snloc )
        end do
! 
        return
        end subroutine filters_for_unstructured_mesh_dg
! 
! 
! 
! 
        subroutine form_dg_filters_from_nx(filt_nxnx, filt_nnx_nosign, &
                                           filt_nnx,  &
                                           nxnx_ele,nxn_ele,snsn, &
                                           dx_face_normal,face_normal, &
                                           ndim,nface,nloc,snloc )
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
! For the arrays:
! filt_nxnx(3,3,1+(ndim-2)*2, ndim, nloc)
! filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! The 3,3 is the dimensions of the filter in 2D. It needs to be a 3x3 array.
! NDIM is the number of dimensions ndim=2 in our case but will change to3 soon.
! nloc is the number of local noes in an element. So we produce a filter for each local node of an element - 4 in our case.
! nsign=1 or 2 and is 1 is we have a positive sign of the component of interest (idim that goes into
! the ndim part of the array) of the velocity. Its 2 if the component is negative. 
        implicit none
        integer, parameter :: nsign=2
        integer, intent( in ) :: nface,ndim,nloc,snloc
        real, intent( in ) :: nxnx_ele(ndim,nloc,nloc),nxn_ele(ndim,nloc,nloc)
        real, intent( in ) :: snsn(snloc,snloc,nface)
        real, intent( in ) :: dx_face_normal(nface),face_normal(ndim,nface)
        real, intent( out ) :: filt_nxnx(3,3,1+(ndim-2)*2, ndim,nloc)
        real, intent( out ) :: filt_nnx_nosign(3,3,1+(ndim-2)*2, ndim,nloc)
        real, intent( out ) :: filt_nnx(3,3,1+(ndim-2)*2, ndim,nsign,nloc)
! local variables...
        integer iloc,jloc,idim,idir, siloc,sjloc
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, i2,j2,k2
        integer iswitch,istart,ifinish
        integer jswitch,jstart,jfinish
        integer kswitch,kstart,kfinish
        integer iface, iloc_d
        integer ii,jj,kk
        integer isig
! 
!        print *,'---ndim,nloc:',ndim,nloc
! 
        filt_nxnx=0.0
        filt_nnx_nosign=0.0
        filt_nnx =0.0
        do kkblock=1,ndim-1
        do jjblock=1,2
        do iiblock=1,2
           iloc=1 + (2-iiblock) +(2-jjblock)*2   + (ndim-2)*(2-kkblock)*4
           print *,'iiblock,jjblock,kkblock,iloc:',iiblock,jjblock,kkblock,iloc
           do kfil=1,ndim-1
           do jfil=1,2
           do ifil=1,2
!              do iidim2=1,2
!              do jjdim2=1,2
!              do kkdim2=1,ndim-1
                 iloc=1 + (2-iiblock) +(2-jjblock)*2   + (ndim-2)*(2-kkblock)*4
!                 floc=iloc
!                 jloc=iiblock +(jjblock-1)*2   + (kkblock-1)*4
                 jloc=ifil +(jfil-1)*2   + (kfil-1)*4
!                 jloc=iidim2 +(jjdim2-1)*2 + (kkdim2-1)*4
                 i=ifil + (iiblock-1)*1
                 j=jfil + (jjblock-1)*1
                 k=kfil + (kkblock-1)*1
                 print *,'i,j,k:',i,j,k
! Volume integrals...
                 filt_nxnx(i,j,k,:,iloc)  = filt_nxnx(i,j,k,:,iloc)  + nxnx_ele(:,iloc,jloc)
                 filt_nnx_nosign(i,j,k,:,iloc) = filt_nnx_nosign(i,j,k,:,iloc) - nxn_ele(:,iloc,jloc)
           end do ! do ifil=1,2
           end do ! do jfil=1,2
           end do ! do kfil=1,ndim-1
        end do ! do iiblock=1,2
        end do ! do jjblock=1,2
        end do ! do kkblock=1,ndim-1
! 
!        print *,'diffusion part (no surface integrals):'
!        do iloc=1,nloc
!           do k=1,1+2*(ndim-2)
!           do j=3,1,-1
!              print *,'j,k,iloc,filt_nxnx(:,j,k,iloc):',j,k,iloc,filt_nxnx(:,j,k,iloc)
!           end do
!           end do
!           print *,' '
!        end do
!        stop 921

        do iface = 1, nface !  Between_Elements_And_Boundary 
           print *,'++++++++++++iface:',iface
! 
! idim is the dimension on which the face is and idir =1 if 1st face of dimension and =2 if second face. 
! iswitch=1 if face is on dimension idim else=0 similarly jswitch associated with idim=2. 
           idim = 1 + (iface-1)/2 
           idir = iface - idim*2 +2

!           iswitch =   1-min(1,max(0,1-idim,idim-1))
           iswitch =   1-min(1,abs(1-idim))
           istart =1*(1-iswitch) + idir*iswitch
           ifinish=2*(1-iswitch) + idir*iswitch

!           jswitch =   1-max(0,2-idim,idim-2)
           jswitch =   1-abs(2-idim)
           jstart =1*(1-jswitch) + idir*jswitch
           jfinish=2*(1-jswitch) + idir*jswitch

           kswitch =   max(0,idim-2)
           kstart =1*(1-kswitch) + idir*kswitch
           kfinish=(ndim-1)*(1-kswitch) + idir*kswitch
                 print *,'idim:',idim
                 print *,'kswitch,jswitch,iswitch,idir:',kswitch,jswitch,iswitch,idir
                 print *,'kstart,kfinish:',kstart,kfinish
                 print *,'jstart,jfinish:',jstart,jfinish
                 print *,'istart,ifinish:',istart,ifinish
! 
           siloc=0
           do kk=kstart,kfinish
           do jj=jstart,jfinish
           do ii=istart,ifinish
              iloc=1+(ii-1) + (jj-1)*2 + (kk-1)*4
              siloc=siloc+1
! 
!              kkblock = 1 + (iloc-1)/4
!              iloc_d  = iloc - (kkblock-1)*4
!              jjblock = 1 + (iloc_d-1)/2
!              iiblock = iloc_d - (jjblock-1)*2

              kkblock = (3-kk)*(ndim-2) + 1*(3-ndim)
              jjblock = 3-jj
              iiblock = 3-ii
                print *,'iloc,siloc:',iloc,siloc
                print *,'iiblock,jjblock,kkblock:',iiblock,jjblock,kkblock
!                
              sjloc=0
              do kfil=kstart,kfinish
              do jfil=jstart,jfinish
              do ifil=istart,ifinish
                 sjloc=sjloc+1
! 
                 i=ifil + (iiblock-1)*1
                 j=jfil + (jjblock-1)*1
                 k=kfil + (kkblock-1)*1
! 
                 i2=ifil + (iiblock-1)*1 + iswitch*(2*idir-3) 
                 j2=jfil + (jjblock-1)*1 + jswitch*(2*idir-3) 
                 k2=kfil + (kkblock-1)*1 + kswitch*(2*idir-3) 

! Surface element contributions for Laplacian...
            print *,'ifil,iiblock,iswitch,idir,sjloc:',ifil,iiblock,iswitch,idir,sjloc
            print *,'i,j,k,i2,j2,k2:',i,j,k,i2,j2,k2
                 filt_nxnx(i,j,k,idim,iloc)  = filt_nxnx(i,j,k,idim,iloc)  &
                                          + snsn(siloc,sjloc,iface)*2.0/dx_face_normal(iface)
            print *,'h1'
                 filt_nxnx(i2,j2,k2,idim,iloc)    = filt_nxnx(i2,j2,k2,idim,iloc)    &
                                          - snsn(siloc,sjloc,iface)*2.0/dx_face_normal(iface)
            print *,'h2'
! Advection...
!                 isig=1 ! positive velocity into element 
                 isig=idir ! positive velocity into element 
                 filt_nnx(i2,j2,k2,idim,isig,iloc) = filt_nnx(i2,j2,k2,idim,isig,iloc) &
                                                + snsn(siloc,sjloc,iface)*face_normal(idim,iface)
            print *,'h3'
!                 isig=2 ! -ve velocity out of element 
                 isig=3-idir ! -ve velocity out of element 
                 filt_nnx(i,j,k,idim,isig,iloc)    = filt_nnx(i,j,k,idim,isig,iloc)    &
                                                + snsn(siloc,sjloc,iface)*face_normal(idim,iface)
            print *,'h4'
              end do ! do ifil=istart,ifinish
              end do ! do jfil=jstart,jfinish
              end do ! do kfil=kstart,kfinish
           end do ! do ii=istart,ifinish
           end do ! do jj=jstart,jfinish
           end do ! do kk=kstart,kfinish
! 
        end do ! do iface = 1, nface 
        print *,'just about to leave form_dg_filters_from_nx'
        return
        end subroutine form_dg_filters_from_nx
! 
! 
! 
! 
        subroutine spacial_tables_for_dg_filters(nxnx_ele, nxn_ele, ml, &
                                                 dx_face_normal, face_normal, snsn,  & 
                                                 x_all, totele, nface,ndim,nloc,snloc)
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        integer, parameter :: nsign=2
        integer, intent( in ) :: totele, nface,ndim,nloc,snloc
        real, intent( in ) :: x_all(ndim,totele*nloc)
        real, intent( out ) :: nxnx_ele(ndim,nloc,nloc,totele), nxn_ele(ndim,nloc,nloc,totele)
        real, intent( out ) :: ml(totele*nloc)
        real, intent( out ) :: dx_face_normal(nface,totele),face_normal(ndim,nface,totele)
        real, intent( out ) :: snsn(snloc,snloc,nface,totele)
! local variables...
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
        real, allocatable :: sn(:,:), snlx(:,:,:)
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:)
        real, allocatable :: nxnx(:,:,:), nnx(:,:,:)
! filters for ann:
!        real, allocatable :: afilt_nxnx(:,:,:,:)
        real, allocatable :: afilt_nnx(:,:,:,:)
        real, allocatable :: x_loc(:,:), xc(:), xsgi(:,:), sx_loc(:,:)
        real, allocatable :: norm(:),snorm(:,:),sdetwei(:),sweigh(:)
        integer npoly,ele_type,ele, iloc,jloc,idim, siloc,sjloc
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, gi, iface
        integer sngi, ngi, max_face_list_no
        integer idir
        integer ii,jj,kk
        integer iswitch, istart, ifinish
        integer jswitch, jstart, jfinish
        integer kswitch, kstart, kfinish
        real sarea
! 
!        print *,'x_all:',x_all
!        stop 2991
!         return
! 
        if(ndim==2) then
! 2d:
           sngi=3; ngi=9
           max_face_list_no=2
        else ! ndim=3
! 3d:
           sngi=9; ngi=27
           max_face_list_no=4
        endif

        allocate(n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi))
        allocate(face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no) )
        allocate(face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) )
        allocate(sn(sngi,snloc), snlx(sngi,ndim-1,snloc)) 
! 
        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
        allocate(nxnx(ndim,nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(x_loc(ndim,nloc),xc(ndim),xsgi(sngi,ndim)) 
        allocate(sx_loc(ndim,snloc))
        allocate(norm(ndim),snorm(sngi,ndim),sdetwei(sngi),sweigh(sngi))

!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        iface=1
        sn(:,:) = face_sn(:,:,iface)
        snlx(:,:,:) = face_snlx(:,:,:,iface)
        sweigh(:)=face_sweigh(:,iface)

        print *,'integrating over volns...'
        print *,'face_sn:',face_sn
        print *,'face_snlx:',face_snlx
        print *,'face_sweigh:',face_sweigh
!         stop 3383
! 
! obtain 
! for filters...
        nxnx_ele=0.0
        nxn_ele=0.0
!        nxn_ele=0.0
        do ele = 1, totele ! VOLUME integral
             print *,'here1'
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             print *,'here2 x_loc:',x_loc
             call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )

             print *,'here3'
             do iloc=1,nloc
                ml((ele-1)*nloc+iloc)=sum( n(:,iloc)*detwei(:) )
             end do
             print *,'here4'
             do iloc=1,nloc
             do jloc=1,nloc
                   do idim=1,ndim
          nxnx_ele(idim,iloc,jloc,ele) = sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
          nxn_ele(idim,iloc,jloc,ele) = sum( nx(:,idim,iloc)*n(:,jloc)*detwei(:) )
                   end do
             end do
             end do
             print *,'here5'
        end do
!         stop 3384
!         return
! 
        print *,'integrating over surfaces...'
! surface integrals... 
        do ele = 1, totele ! Surface integral
            ! for copy local memory copying...
           x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
           do idim=1,ndim
              xc(idim)=sum(x_loc(idim,:))/real(nloc) 
           end do
           do iface = 1, nface !  Between_Elements_And_Boundary 
                print *,'---------iface=',iface
! 
! idim is the dimension on which the face is and idir =1 if 1st face of dimension and =2 if second face. 
! iswitch=1 if face is on dimension idim else=0 similarly jswitch associated with idim=2. 
                 idim = 1 + (iface-1)/2 
                 idir = iface - idim*2 +2

!                 iswitch =   1-min(1,max(0,1-idim,idim-1))
                 iswitch =   1-min(1,abs(1-idim))
                 istart =1*(1-iswitch) + idir*iswitch
                 ifinish=2*(1-iswitch) + idir*iswitch

!                 jswitch =   1-max(0,2-idim,idim-2)
                 jswitch =   1-abs(2-idim)
                 jstart =1*(1-jswitch) + idir*jswitch
                 jfinish=2*(1-jswitch) + idir*jswitch

                 kswitch =   max(0,idim-2)
                 kstart =1*(1-kswitch) + idir*kswitch
                 kfinish=(ndim-1)*(1-kswitch) + idir*kswitch
                 print *,'idim:',idim
                 print *,'kswitch,jswitch,iswitch,idir:',kswitch,jswitch,iswitch,idir
                 print *,'kstart,kfinish:',kstart,kfinish
                 print *,'jstart,jfinish:',jstart,jfinish
                 print *,'istart,ifinish:',istart,ifinish
                 siloc=0
                 do kk=kstart,kfinish
                 do jj=jstart,jfinish
                 do ii=istart,ifinish
                    iloc=1+(ii-1) + (jj-1)*2 + (kk-1)*4
                    siloc=siloc+1
                     print *,'iloc,siloc=',iloc,siloc
                    sx_loc(:,siloc) = x_loc(:,iloc)
                     print *,'h3'
                 end do
                 end do
                 end do
!               stop 28

!                 s_list_no = face_list_no( iface, ele) 
!                 sn2 =face_sn2(:,:,s_list_no)

! nn for element ele
!                 snorm(:,ndim+1)=0.0
!            stop 3103
                 xsgi = 0.0
                 do siloc=1,snloc ! use all of the nodes not just the surface nodes. 
                    do idim=1,ndim
                       xsgi(:,idim)  = xsgi(:,idim)  + sn(:,siloc)*sx_loc(idim,siloc) 
!                       print *,'idim,siloc,sx_loc(idim,siloc):',idim,siloc,sx_loc(idim,siloc)
                    end do
                 end do
                     print *,'h4'
                 do idim=1,ndim
                    norm(idim) = sum(xsgi(:,idim))/real(sngi) - xc(idim)
                 end do
!                     print *,'h5'
!                  print *,'nloc, sngi, ndim:',nloc, sngi, ndim
!                  print *,'sx_loc:',sx_loc
!                  print *,'sn:',sn
!                  print *,'snlx:',snlx
!                  print *,'sweigh:',sweigh
!                 do siloc=1,snloc 
!                    do idim=1,ndim-1
!       print *,'snlx(:,idim,siloc):',idim,siloc,snlx(:,idim,siloc)
!                    end do
!                 end do
!            stop 3104
! start to do the integration...
                 call det_snlx_all( snloc, sngi, ndim-1, ndim, sx_loc, sn, snlx, &
                                    sweigh, sdetwei, sarea, snorm, norm )
!         if(iface==3) return ! OK
!             print *,'sdetwei:',sdetwei
!             print *,'sarea:',sarea
!             print *,'snorm:',snorm
!             print *,'norm:',norm
!            stop 3105
!                     print *,'h6'
!         if(iface==3) return ! ok
                 face_normal(:,iface,ele)=snorm(1,:)
!         if(iface==3) return ! not ok
                 dx_face_normal(iface,ele)=sarea**(1./real(ndim)) 
!         if(iface==3) return ! not ok
                 do siloc=1,snloc
                     do sjloc=1,snloc
                        snsn(siloc,sjloc,iface,ele) = sum( sdetwei(:)*sn(:,siloc)*sn(:,sjloc) )
!                        do idim=1,ndim
!                           snsn_normal(idim,siloc,sjloc,iface,ele) = sum( snorm(:,idim)*sdetwei(:)*sn(:,siloc)*sn(:,sjloc) )
!                        end do
                     end do
                  end do
! Put into global vector...
! Integrate both sides...
!            stop 3123
!         if(iface==3) return
           end do ! do iface = 1, nface !  Between_Elements_And_Boundary 
        end do ! do ele = 1, totele ! Surface integral
! 
!         stop 382
        return
        end subroutine spacial_tables_for_dg_filters
! 
! 
! 
! 
        subroutine on_fly_spacial_tables_for_dg_filters(a_filter, ml, source_vec, &
                                                 source,sigma,kdiff, u, dist_ele_face, x_loc,   & 
                                                 n, nlx, weight, sn, snlx, sweigh, &
                                                 nface,ndim,nloc,snloc,ngi,sngi)
        implicit none
! integers representing the length of arrays...
! totele=no of elements,nloc=no of nodes per element, totele_nloc=totele*nloc
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
        integer, parameter :: nsign=2
        integer, intent( in ) :: nface,ndim,nloc,snloc,ngi,sngi
        real, intent( in ) :: source(nloc),sigma(nloc),kdiff(ndim,nloc), u(ndim,nloc)
        real, intent( in ) :: dist_ele_face(nface), x_loc(ndim,nloc)
        real, intent( in ) :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real, intent( in ) :: sn(sngi,snloc), snlx(sngi,ndim-1,snloc), sweigh(sngi)
        real, intent( out ) :: a_filter(3,3,ndim-1,nloc), ml(nloc), source_vec(nloc)
! local variables - put on the stack for speed...
        real nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim)
        real xc(ndim),sx_loc(ndim,snloc)
        real sourcegi(ngi),sigmagi(ngi),kdiffgi(ngi,ndim),ugi(ngi,ndim),dudxgi(ngi,ndim)
! 
        real norm(ndim),snorm(sngi,ndim),sdetwei(sngi)
! 
        real skdiff(ndim,snloc),su(ndim,snloc)
        real kdiffsgi(sngi,ndim),xsgi(sngi,ndim),usgi(sngi,ndim),income(sngi)
        integer iloc,jloc,idim,jdim,idir, siloc,sjloc
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j,k, i2,j2,k2
        integer iswitch,istart,ifinish
        integer jswitch,jstart,jfinish
        integer kswitch,kstart,kfinish
        integer gi, sgi, iface, iloc_d
        integer ii,jj,kk
        real sarea, inv_dis, cont_diffusion, cont_advection_in, cont_advection_out
! 
! 
        a_filter=0.0
! VOLUME integral
        call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )

        do gi=1,ngi
           sourcegi(gi)=sum( n(gi,:)*source(:) )
           sigmagi(gi)=sum( n(gi,:)*sigma(:) )
           do idim=1,ndim
              kdiffgi(gi,idim)=sum( n(gi,:)*kdiff(idim,:) )
              ugi(gi,idim)   = sum( n(gi,:)*u(idim,:) )
              dudxgi(gi,idim)= sum( nx(gi,idim,:)*u(idim,:) )
           end do
        end do
        do iloc=1,nloc
            ml(iloc)=sum( n(:,iloc)*detwei(:) )
            source_vec(iloc)=sum( n(:,iloc)*sourcegi(:)*detwei(:) )
        end do
        do kkblock=1,ndim-1
        do jjblock=1,2
        do iiblock=1,2
           iloc=1 + (2-iiblock) +(2-jjblock)*2   + (ndim-2)*(2-kkblock)*4
           do kfil=1,ndim-1
           do jfil=1,2
           do ifil=1,2
              jloc=ifil +(jfil-1)*2   + (kfil-1)*4
! 
              i=ifil + (iiblock-1)*1
              j=jfil + (jjblock-1)*1
              k=kfil + (kkblock-1)*1
              a_filter(i,j,k,iloc)=a_filter(i,j,k,iloc) + & 
                     sum( n(:,iloc)*sigmagi(:)*n(:,jloc)*detwei(:) )  ! sbsoption
              do idim=1,ndim
                 a_filter(i,j,k,iloc)=a_filter(i,j,k,iloc) + & 
                 sum( (  nx(:,idim,iloc)*kdiffgi(:,idim)*nx(:,idim,jloc)  & ! diffusion
                       +(nx(:,idim,iloc)*ugi(:,idim)+n(:,iloc)*dudxgi(:,idim))*n(:,jloc) & ! advection
                               )*detwei(:) )
              end do
           end do ! do ifil=1,2
           end do ! do jjblock=1,2
           end do ! do kkblock=1,ndim-1
        end do ! do iiblock=1,2
        end do ! do jjblock=1,2
        end do ! do kkblock=1,ndim-1
! 
! surface integrals... 
        do idim=1,ndim
           xc(idim)=sum(x_loc(idim,:))/real(nloc) 
        end do

        do iface = 1, nface !  Between_Elements_And_Boundary 
! 
! idim is the dimension on which the face is and idir =1 if 1st face of dimension and =2 if second face. 
! iswitch=1 if face is on dimension idim else=0 similarly jswitch associated with idim=2. 
           idim = 1 + (iface-1)/2 
           idir = iface - idim*2 +2

!           iswitch =   1-min(1,max(0,1-idim,idim-1))
           iswitch =   1-min(1,abs(1-idim))
           istart =1*(1-iswitch) + idir*iswitch
           ifinish=2*(1-iswitch) + idir*iswitch

!           jswitch =   1-max(0,2-idim,idim-2)
           jswitch =   1-abs(2-idim)
           jstart =1*(1-jswitch) + idir*jswitch
           jfinish=2*(1-jswitch) + idir*jswitch

           kswitch =   max(0,idim-2)
           kstart =1*(1-kswitch) + idir*kswitch
           kfinish=(ndim-1)*(1-kswitch) + idir*kswitch
! 
           siloc=0
           do kk=kstart,kfinish
           do jj=jstart,jfinish
           do ii=istart,ifinish
              iloc=1+(ii-1) + (jj-1)*2 + (kk-1)*4
              siloc=siloc+1
              sx_loc(:,siloc) = x_loc(:,iloc)
              skdiff(:,siloc) = kdiff(:,iloc)
              su(:,siloc) = u(:,iloc)
           end do
           end do
           end do
! 
           do sgi=1,sngi
              do jdim=1,ndim
                 kdiffsgi(sgi,jdim)=sum( sn(sgi,:)*skdiff(jdim,:) )
                 usgi(sgi,jdim) = sum( sn(sgi,:)*su(jdim,:) )
                 xsgi(sgi,jdim) = sum( sn(sgi,:)*sx_loc(jdim,:) )  
              end do
           end do
           do jdim=1,ndim
              norm(jdim) = sum(xsgi(:,jdim))/real(sngi) - xc(jdim)
           end do
! start to do the integration...
           call det_snlx_all( snloc, sngi, ndim-1, ndim, sx_loc, sn, snlx, &
                                    sweigh, sdetwei, sarea, snorm, norm )
           inv_dis = 1.0/dist_ele_face(iface)
           siloc=0
           do kk=kstart,kfinish
           do jj=jstart,jfinish
           do ii=istart,ifinish
              iloc=1+(ii-1) + (jj-1)*2 + (kk-1)*4
              siloc=siloc+1
! 
!              kkblock = 1 + (iloc-1)/4
!              iloc_d  = iloc - (kkblock-1)*4
!              jjblock = 1 + (iloc_d-1)/2
!              iiblock = iloc_d - (jjblock-1)*2

              kkblock = (3-kk)*(ndim-2) + 1*(3-ndim)
              jjblock = 3-jj
              iiblock = 3-ii
!                
              sjloc=0
              do kfil=kstart,kfinish
              do jfil=jstart,jfinish
              do ifil=istart,ifinish
                 sjloc=sjloc+1
! 
                 i=ifil + (iiblock-1)*1
                 j=jfil + (jjblock-1)*1
                 k=kfil + (kkblock-1)*1
! 
                 i2=ifil + (iiblock-1)*1 + iswitch*(2*idir-3)
                 j2=jfil + (jjblock-1)*1 + jswitch*(2*idir-3)
                 k2=kfil + (kkblock-1)*1 + kswitch*(2*idir-3)
! nn for element ele
!                 snorm(:,ndim+1)=0.0
! Integrate both sides...
                 cont_diffusion = 2.*inv_dis*sum( sn(:,siloc)*kdiffsgi(:,idim)*sn(:,sjloc) ) ! diffusion
                 a_filter(i,j,k,iloc)   =a_filter(i,j,k,iloc)    + cont_diffusion
                 a_filter(i2,j2,k2,iloc)=a_filter(i2,j2,k2,iloc) - cont_diffusion ! contribution on otherside of element
                 do sgi=1,sngi
                    income(sgi) = 0.5*sign(1.0, sum(-usgi(sgi,:)*snorm(sgi,:)))+0.5
                 end do
                 do jdim=1,ndim ! advection
                    cont_advection_in = sum( sn(:,siloc)*usgi(:,jdim)*snorm(:,jdim)*income(:)*sn(:,sjloc)*detwei(:) )
                    cont_advection_out= sum( sn(:,siloc)*usgi(:,jdim)*snorm(:,jdim)*(1.-income(:))*sn(:,sjloc)*detwei(:) )
                    a_filter(i,j,k,iloc)   =a_filter(i,j,k,iloc)    + cont_advection_out
                    a_filter(i2,j2,k2,iloc)=a_filter(i2,j2,k2,iloc) + cont_advection_in
                 end do
              end do ! do ifil=istart,ifinish
              end do ! do jfil=jstart,jfinish
              end do ! do kfil=kstart,kfinish
           end do ! do ii=istart,ifinish
           end do ! do jj=jstart,jfinish
           end do ! do kk=kstart,kfinish
! 
        end do ! do iface = 1, nface 
!         stop 382
        return
        end subroutine on_fly_spacial_tables_for_dg_filters
! 
! 
! 
! 
! The call from python looks like: 
! cube_sn_weight, cube_sn_direction = cube_sn_quadrature(nx_cube, ny_cube)
        subroutine cube_sn_quadrature(cube_sn_weight, cube_sn_direction, nx_cube, ny_cube)
! This subroutine caclculates the Sn quadrature set. 
! It gives the direction of each Sn ordinate: cube_sn_direction(ix_cube, iy_cube, iface_cube) 
! in which ix_cube, iy_cube defines the cell on one of the 6 faces iface_cube on the cube-sphere. 
! cube_sn_weight contains the area associated with the patch on the unit sphere of 
! this Sn quadrature set. 
! nx_cube, ny_cube are the dimensions of the regular grid on each face of the cube  
! and for a multi-grid method it is suggested that they are a power of 2 so that the mult-grid 
! can coursen to one cell which results in 6 cells in total - one on each of the 6 faces. 
! The faces look like:
!        !---------!
!        !         !\
!        !   5     ! \
!        !         !  \
!        !---------! 2!
!         \    3    \ !
!          \---------\!
! or in the xz plane:
!       !---4---!
!       !       !
!       1       2
!       !       !
!       !---3---!
! or in the yz plane:
!       !---6---!
!       !       !
!       1       2
!       !       !
!       !---5---!
! 
        implicit none
        integer, intent( in ) :: nx_cube, ny_cube
        integer, parameter :: ndim=3, nloc=8, snloc=4, nface_cube=6 
        real, intent( out ) :: cube_sn_weight(nx_cube, ny_cube, nface_cube)
        real, intent( out ) :: cube_sn_direction(ndim,nx_cube, ny_cube, nface_cube)
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
! nloc=no of nodes per element. 
! local variables...
        real, parameter :: pi = 3.14159265358
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
        real, allocatable :: sn(:,:), snlx(:,:,:)
        real, allocatable :: norm(:),snorm(:,:),sdetwei(:),sweigh(:)
        integer npoly,ele_type,idim, siloc, iface, nface, ix_cube, iy_cube, iface_cube
        integer sngi, ngi, max_face_list_no
        real sarea, dx, rnorm, sarea_sum
        real direction(ndim), x_corn(ndim), sx_loc(ndim,snloc)
!      
! 3d:
        sngi=9
        ngi=27
        max_face_list_no=4
        nface=6

        allocate(n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi))
        allocate(face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no) )
        allocate(face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) )
        allocate(sn(sngi,snloc), snlx(sngi,ndim-1,snloc)) 
! 
        allocate(norm(ndim),snorm(sngi,ndim),sdetwei(sngi),sweigh(sngi))

!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        iface=1
        sn(:,:) = face_sn(:,:,iface)
        snlx(:,:,:) = face_snlx(:,:,:,iface)
        sweigh(:)=face_sweigh(:,iface)

! 
! surface integrals... 
        dx=2.0/real(nx_cube)
! do this for just the top surface of the sphere then work out the 
! sn wieghts and directions for the other faces. 

        do iy_cube=1,ny_cube
        do ix_cube=1,nx_cube
!           ele = (iy_oct-1)*nx_oct + ix_oct
            ! for copy local memory copying...
           x_corn(1)=real(ix_cube-1)*dx - 1.0
           x_corn(2)=real(iy_cube-1)*dx - 1.0
! define cornders of the square...
           sx_loc(1,1) = x_corn(1)
           sx_loc(2,1) = x_corn(2)

           sx_loc(1,2) = x_corn(1) + dx
           sx_loc(2,2) = x_corn(2)

           sx_loc(1,3) = x_corn(1) 
           sx_loc(2,3) = x_corn(2) + dx

           sx_loc(1,4) = x_corn(1) + dx
           sx_loc(2,4) = x_corn(2) + dx
! 
           sx_loc(3,:) = 1.0
! now project onto unit sphere
           do siloc=1,snloc
              rnorm=sqrt(sum(sx_loc(:,siloc)**2))
              sx_loc(:,siloc)=sx_loc(:,siloc)/rnorm
           end do
! rotate according to iface_cube
! normal=(-1,0,0) to cube face...
           norm(:) = 0.0
           norm(3) = 1.0
! 
           iface = 1
           iface_cube=4
                   
!                 s_list_no = face_list_no( iface, ele) 
!           sn = face_sn(:,:,iface)
!           snlx(:,:,:) = face_snlx(:,:,:,iface)
!                 sn2 =face_sn2(:,:,s_list_no)
! start to do the integration...
           call det_snlx_all( snloc, sngi, ndim-1, ndim, sx_loc, sn, snlx, &
                              sweigh, sdetwei, sarea, snorm, norm )
           cube_sn_weight(ix_cube,iy_cube,iface_cube) = sarea
           do idim=1,ndim
              direction(idim) = sum(sx_loc(idim,:))/real(snloc) 
           end do
! make sure the sn direction has unit length
           rnorm=sqrt(sum(direction(:)**2))
           cube_sn_direction(:,ix_cube,iy_cube,iface_cube) = direction(:)/rnorm

! Put into global vector...
        end do ! do ix_oct=1,ny_oct
        end do ! do iy_oct=1,ny_oct

! norm form the sn quadrature on the 6 faces of the cube. 
        do iface_cube=1,nface_cube
           cube_sn_direction(:,:,:,iface_cube) = cube_sn_direction(:,:,:,4) 
! normal=(-1,0,0) to cube face...
           if(iface_cube==1) then ! swap x and z over and make x -ve
              cube_sn_direction(1,:,:,iface_cube) = -cube_sn_direction(3,:,:,4) 
              cube_sn_direction(3,:,:,iface_cube) = cube_sn_direction(1,:,:,4) 
           endif
! normal=(1,0,0) to cube face...
           if(iface_cube==2) then ! swap x and z over
              cube_sn_direction(1,:,:,iface_cube) = cube_sn_direction(3,:,:,4) 
              cube_sn_direction(3,:,:,iface_cube) = cube_sn_direction(1,:,:,4) 
           endif
! normal=(0,0,-1) to cube face...
           if(iface_cube==3) then ! change sign of z coord
              cube_sn_direction(3,:,:,iface_cube) = -cube_sn_direction(3,:,:,4) 
           endif
! normal=(0,0,1) to cube face...
           if(iface_cube==4) then ! iface_cube==4 do nothing.
           endif
! normal=(0,-1,0) to cube face...
           if(iface_cube==5) then ! swap y and z over and make y -ve
              cube_sn_direction(2,:,:,iface_cube) = -cube_sn_direction(3,:,:,4) 
              cube_sn_direction(3,:,:,iface_cube) = cube_sn_direction(2,:,:,4) 
           endif
! normal=(0,1,0) to cube face...
           if(iface_cube==6) then ! swap y and z over
              cube_sn_direction(2,:,:,iface_cube) = cube_sn_direction(3,:,:,4) 
              cube_sn_direction(3,:,:,iface_cube) = cube_sn_direction(2,:,:,4) 
           endif
           cube_sn_weight(:,:,iface_cube) = cube_sn_weight(:,:,4) 
        end do ! do iface_oct=1,nface_oct
! r-normalise so the area is the same as the unit sphere. 
        sarea_sum = sum(cube_sn_weight) 
        cube_sn_weight = cube_sn_weight * 4.0*pi/sarea_sum
! 
!         stop 382
        return
        end subroutine cube_sn_quadrature
! 
! 
! 
! 
        subroutine file_generator4octant_sn_quadrature
! this subroutine outputs a file containing the quadrature directions and 
! weights associated with the discrete ordinate discretisation of the unit sphere. 
! nx_cube, ny_cube are the number of cells across and up for each of 
! the eight patches on the octant that produces a discretisation of the unit sphere.
! CHANGE nx_cube, ny_cube as necessary to increase the quadrature points. 
        integer, parameter :: nx_cube=4, ny_cube=4  
        integer, parameter :: ndim=3, nloc=8, snloc=4, nface_cube=8 
        integer iface_cube, idim, ix_cube, iy_cube
        real, allocatable :: cube_sn_weight(:, :, :)
        real, allocatable :: cube_sn_direction(:,:, :, :)

        allocate(cube_sn_weight(nx_cube, ny_cube, nface_cube))
        allocate(cube_sn_direction(ndim,nx_cube, ny_cube, nface_cube))
! 
        call octant_sn_quadrature(cube_sn_weight, cube_sn_direction, nx_cube, ny_cube)
! 
        open(27, file='octant_sn_quadrature.csv')
        write(27,*) 'This outputs a file containing the quadrature directions and '
        write(27,*) 'weights associated with the discrete ordinate discretisation of the unit sphere.' 
        write(27,*) 'nx_cube, ny_cube are the number of cells across and up for each of '
        write(27,*) 'the eight patches on the octant that produces a discretisation of the unit sphere.'
        write(27,*)
        write(27,*) 'nface_cube, ndim, nx_cube, ny_cube'
        write(27,*) 'do iface_cube=1,nface_cube'
        write(27,*) '      do iy_cube = 1, ny_cube'
        write(27,*) '         write(27,*) cube_sn_weight(:,iy_cube,iface_cube) '
        write(27,*) '      end do'
        write(27,*) 'end do ! do iface_cube=1,nface_cube'
        write(27,*) 
        write(27,*) 'do iface_cube=1,nface_cube'
        write(27,*) '   do idim=1,ndim'
        write(27,*) '      do iy_cube = 1, ny_cube'
        write(27,*) '         write(27,*) cube_sn_direction(idim,:,iy_cube,iface_cube) '
        write(27,*) '      end do'
        write(27,*) '   end do ! do idim=1,ndim'
        write(27,*) 'end do ! do iface_cube=1,nface_cube'
        write(27,*)
        write(27,*) nface_cube, ndim, nx_cube, ny_cube
        do iface_cube=1,nface_cube
              do iy_cube = 1, ny_cube
                 write(27,*) cube_sn_weight(:,iy_cube,iface_cube) 
              end do
        end do ! do iface_cube=1,nface_cube
        write(27,*)
        do iface_cube=1,nface_cube
           do idim=1,ndim
              do iy_cube = 1, ny_cube
                 write(27,*) cube_sn_direction(idim,:,iy_cube,iface_cube) 
              end do
           end do ! do idim=1,ndim
        end do ! do iface_cube=1,nface_cube
        close(27)
        return
        end subroutine file_generator4octant_sn_quadrature
! 
! 
! 
! 
! The call from python looks like: 
! cube_sn_weight, cube_sn_direction = octant_sn_quadrature(nx_cube, ny_cube)
        subroutine octant_sn_quadrature(cube_sn_weight, cube_sn_direction, nx_cube, ny_cube)
! This subroutine caclculates the Sn quadrature set. 
! It gives the direction of each Sn ordinate: octant_sn_direction(ix_cube, iy_cube, iface_cube) 
! in which ix_cube, iy_cube defines the cell on one of the 6 faces iface_cube on the octant-sphere. 
! cube_sn_weight contains the area associated with the patch on the unit sphere of 
! this Sn quadrature set. 
! nx_cube, ny_cube are the dimensions of the regular grid on each face of the cube  
! and for a multi-grid method it is suggested that they are a power of 2 so that the mult-grid 
! can coursen to one cell which results in 8 cells in total - one on each of the 8 faces. 
! The faces look like at the top of the sphere:
!          o     
!        / ! \     
!      /   !   \    
!    /  3  !  4  \   
!   o------o------o
!    \  1  !  2  /     
!      \   !   /   
!        \ ! /   
!          o
! and at the bottom of the sphere - still looking from the top but underneath:
!          o     
!        / ! \     
!      /   !   \    
!    /  7  !  8  \   
!   o------o------o
!    \  5  !  6  /     
!      \   !   /   
!        \ ! /   
!          o
        implicit none
        integer, intent( in ) :: nx_cube, ny_cube
        integer, parameter :: ndim=3, nloc=8, snloc=4, nface_cube=8 
        real, intent( out ) :: cube_sn_weight(nx_cube, ny_cube, nface_cube)
        real, intent( out ) :: cube_sn_direction(ndim,nx_cube, ny_cube, nface_cube)
! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces.
! ndim=no of dimensions - including time possibly, nface=no of faces of each elemenet, nc no of fields to solve for.
! nloc=no of nodes per element. 
! local variables...
        real, parameter :: pi = 3.14159265358
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
        real, allocatable :: sn(:,:), snlx(:,:,:)
        real, allocatable :: norm(:),snorm(:,:),sdetwei(:),sweigh(:)
        real, allocatable :: sx_loc_sgi(:,:)
        integer npoly,ele_type,idim, siloc, iface, nface, ix_cube, iy_cube, iface_cube
        integer sngi, ngi, max_face_list_no, sgi, icorn
        real sarea, dx, rnorm, sarea_sum
        real direction(ndim), x_corn(ndim), sx_loc(ndim,snloc)
        real lx(4), ly(4), lz(4), x_interp_corn(ndim,4)
!      
! 3d:
        sngi=9
        ngi=27
        max_face_list_no=4
        nface=6

        allocate(n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi))
        allocate(face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no) )
        allocate(face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) )
        allocate(sn(sngi,snloc), snlx(sngi,ndim-1,snloc)) 
! 
        allocate(norm(ndim),snorm(sngi,ndim),sdetwei(sngi),sweigh(sngi))

        allocate(sx_loc_sgi(ndim,sngi))

!         stop 3382
        npoly=1
        ele_type=1

! form the shape functions...
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        iface=1
        sn(:,:) = face_sn(:,:,iface)
        snlx(:,:,:) = face_snlx(:,:,:,iface)
        sweigh(:)=face_sweigh(:,iface)

! 
! surface integrals... 
        dx=1.0/real(nx_cube)
! do this for just the top surface of the sphere then work out the 
! sn wieghts and directions for the other faces. 
! interpolations corners
! local nodes...
!       3-------4     
!       !       !
!       !       !
!       !       !
!       1-------2
! This is mapped to a triangle with a collapsed node (1 and 3): 
!       4     
!       ! \     
!       !   \    
!       !     \   
!      1,3------2
! With nodal coordinates...
!     (0,1)     
!       ! \     
!       !   \    
!       !     \   
!     (0,0)----(1,0)
! the cells are numbered in the non-distorted system...
!        1,4   2,4   3,4   4,4
!        1,3   2,3   3,3   4,3
!        1,2   2,2   3,2   4,2
!        1,1   2,1   3,1   4,1
! 1st node...
        x_interp_corn(1,1)=0.0
        x_interp_corn(2,1)=0.0
        x_interp_corn(3,1)=1.0
! 2nd node...
        x_interp_corn(1,2)=1.0
        x_interp_corn(2,2)=0.0
        x_interp_corn(3,2)=0.0
! 3rd node (collapsed node same as node 1)...
        x_interp_corn(1,3)=0.0
        x_interp_corn(2,3)=0.0
        x_interp_corn(3,3)=1.0
! 4th node...
        x_interp_corn(1,4)=0.0
        x_interp_corn(2,4)=1.0
        x_interp_corn(3,4)=0.0
! 
        do iy_cube=1,ny_cube
        do ix_cube=1,nx_cube
!           ele = (iy_oct-1)*nx_oct + ix_oct
            ! for copy local memory copying...
           x_corn(1)=real(ix_cube-1)*dx 
           x_corn(2)=real(iy_cube-1)*dx 
! define cornders of the square...
           lx(1) = x_corn(1)
           ly(1) = x_corn(2)

           lx(2) = x_corn(1) + dx
           ly(2) = x_corn(2)

           lx(3) = x_corn(1) 
           ly(3) = x_corn(2) + dx

           lx(4) = x_corn(1) + dx
           ly(4) = x_corn(2) + dx

           lz(:)=1.0-lx(:)

! now linearly interpolate corner nodes
           do icorn=1,4
              sx_loc(1:3,icorn) = (1.0-lx(icorn))*(1.0-ly(icorn))*x_interp_corn(1:3,1) &
                                +      lx(icorn) *(1.0-ly(icorn))*x_interp_corn(1:3,2) &
                                + (1.0-lx(icorn))*ly(icorn)      *x_interp_corn(1:3,3) &
                                +      lx(icorn) *ly(icorn)      *x_interp_corn(1:3,4)
           end do
! now project onto unit sphere
           do siloc=1,snloc
              rnorm=sqrt(sum(sx_loc(:,siloc)**2))
              sx_loc(:,siloc)=sx_loc(:,siloc)/rnorm
           end do
! rotate according to iface_cube
! normal=(-1,0,0) to cube face...
           norm(:) = 0.0
           norm(3) = 1.0
! 
           iface = 1
           iface_cube=4
                   
!                 s_list_no = face_list_no( iface, ele) 
!           sn = face_sn(:,:,iface)
!           snlx(:,:,:) = face_snlx(:,:,:,iface)
!                 sn2 =face_sn2(:,:,s_list_no)
! start to do the integration...
           call det_snlx_all( snloc, sngi, ndim-1, ndim, sx_loc, sn, snlx, &
                              sweigh, sdetwei, sarea, snorm, norm )
           cube_sn_weight(ix_cube,iy_cube,iface_cube) = sarea
           do idim=1,ndim
              do sgi=1,sngi
                 sx_loc_sgi(idim,sgi) = sum(sn(sgi,:)*sx_loc(idim,:)) 
              end do
           end do
           do idim=1,ndim
              direction(idim) = sum(sx_loc_sgi(idim,:)*sdetwei(:))/sum(sdetwei(:)) 
           end do
! make sure the sn direction has unit length
           rnorm=sqrt(sum(direction(:)**2))
           cube_sn_direction(:,ix_cube,iy_cube,iface_cube) = direction(:)/rnorm

! Put into global vector...
        end do ! do ix_oct=1,ny_oct
        end do ! do iy_oct=1,ny_oct

! norm form the sn quadrature on the 6 faces of the cube. 
        do iface_cube=1,nface_cube
           cube_sn_direction(:,:,:,iface_cube) = cube_sn_direction(:,:,:,4) 
! normal=(-1,0,0) to cube face...
           if(iface_cube==1) then ! swap x and z over and make x -ve
              cube_sn_direction(1,:,:,iface_cube) = -cube_sn_direction(1,:,:,4)
              cube_sn_direction(2,:,:,iface_cube) = -cube_sn_direction(2,:,:,4)  
              cube_sn_direction(3,:,:,iface_cube) = +cube_sn_direction(3,:,:,4) 
           endif
! normal=(1,0,0) to cube face...
           if(iface_cube==2) then ! swap x and z over
              cube_sn_direction(1,:,:,iface_cube) = +cube_sn_direction(1,:,:,4)
              cube_sn_direction(2,:,:,iface_cube) = -cube_sn_direction(2,:,:,4)  
              cube_sn_direction(3,:,:,iface_cube) = +cube_sn_direction(3,:,:,4) 
           endif
! normal=(0,0,-1) to cube face...
           if(iface_cube==3) then ! change sign of z coord
              cube_sn_direction(1,:,:,iface_cube) = -cube_sn_direction(1,:,:,4)
              cube_sn_direction(2,:,:,iface_cube) = +cube_sn_direction(2,:,:,4)  
              cube_sn_direction(3,:,:,iface_cube) = +cube_sn_direction(3,:,:,4) 
           endif
! normal=(0,0,1) to cube face...
           if(iface_cube==4) then ! iface_cube==4 do nothing.
           endif
! normal=(0,-1,0) to cube face...
           cube_sn_weight(:,:,iface_cube) = cube_sn_weight(:,:,4) 
        end do ! do iface_oct=1,nface_oct
! bottom half of the unit sphere.  
        cube_sn_direction(1,:,:,5:8) = +cube_sn_direction(1,:,:,1:4) 
        cube_sn_direction(2,:,:,5:8) = +cube_sn_direction(2,:,:,1:4) 
        cube_sn_direction(3,:,:,5:8) = -cube_sn_direction(3,:,:,1:4) 
! 
! r-normalise so the area is the same as the unit sphere. 
        sarea_sum = sum(cube_sn_weight) 
        cube_sn_weight = cube_sn_weight * 4.0*pi/sarea_sum
! 
!         stop 382
        return
        end subroutine octant_sn_quadrature
! 
! 
! 
! 
! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
      subroutine sfc_mapping_to_sfc_matrix(a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
                                     a, b, ml, &
                                     fina,cola, sfc_node_ordering, ncola, &
                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
! this subroutine finds the space filling curve representation of matrix eqns A T=b 
! - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
! It also puts the vector b in space filling curve ordering. 
! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
! which are stored in b_sfc, ml_sfc. 
        implicit none
! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
! are nlevel grids from course to fine. 
! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
! sfc_node_ordering(i_sfc_order)=fem node number. Here i_sfc_order is the number of the node meansured along 
! the space filling curve trajectory. 
! nonods=number of finite element nodes in the mesh.
! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
! call in python: nlevel = calculate_nlevel_sfc(nonods)
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids, max_nlevel
        real, intent( out ) :: a_sfc(3,max_nonods_sfc_all_grids), b_sfc(max_nonods_sfc_all_grids), &
                               ml_sfc(max_nonods_sfc_all_grids) 
        real, intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel), nlevel
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
! local variables...
        integer i, count, nodj, prev_nodi_sfc, next_nodi_sfc, nodi_sfc, ilevel
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
! 
        print *,'dont use - use the best version' 
        stop 2231
! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
! form SFC matrix...
        a_sfc(:,:)=0.0
        b_sfc(:)=0.0; b_sfc(1:nonods)=b(1:nonods)
        ml_sfc(:)=0.0; ml_sfc(1:nonods)=ml(1:nonods)
! 
        do i=1,nonods
           prev_nodi_sfc = 0
           next_nodi_sfc = 0
           if(i.ne.1) prev_nodi_sfc = sfc_node_ordering(i-1)
           if(i.ne.nonods) next_nodi_sfc = sfc_node_ordering(i+1)
           nodi_sfc=sfc_node_ordering(i)
           do count=fina(nodi_sfc),fina(nodi_sfc+1)-1
              nodj=cola(count)
              if(nodj==prev_nodi_sfc) then
                 a_sfc(1,nodi_sfc)=a_sfc(1,nodi_sfc)+a(count)
              else if(nodj==next_nodi_sfc) then
                 a_sfc(3,nodi_sfc)=a_sfc(3,nodi_sfc)+a(count)
              else 
                 a_sfc(2,nodi_sfc)=a_sfc(2,nodi_sfc)+a(count) ! diagonal
              endif
           end do
        end do
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
! 
! coarsen...
        do ilevel=2,nlevel
           sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
           if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
           sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)

           call map_sfc_fine_grid_2_course_grid_vec( &
                                         ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           call map_sfc_fine_grid_2_course_grid_vec( &
                                         b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                                         b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
           sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
           fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
           print *,'run out of memory here stopping'
           stop 2822
        endif
        return 
        end subroutine sfc_mapping_to_sfc_matrix
! 
! 
! 
! 
        subroutine map_sfc_course_grid( a_sfc_course,sfc_nonods_course, a_sfc_fine,sfc_nonods_fine)
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real, intent( out ) :: a_sfc_course(3,sfc_nonods_course)
        real, intent( in ) :: a_sfc_fine(3,sfc_nonods_fine)
! local variables...
        integer i_short, i_long
! 
        a_sfc_course(:,:)=0.0
        do i_short=1,sfc_nonods_course
           i_long=(i_short-1)*2 + 1
           a_sfc_course(1,i_short)=a_sfc_course(1,i_short)+a_sfc_fine(1,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(2,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(3,i_long)

           i_long=(i_short-1)*2 + 2
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(1,i_long)
           a_sfc_course(2,i_short)=a_sfc_course(2,i_short)+a_sfc_fine(2,i_long)
           a_sfc_course(3,i_short)=a_sfc_course(3,i_short)+a_sfc_fine(3,i_long)
        end do
        
        return 
        end subroutine map_sfc_course_grid
! 
! 
! 
! 
        subroutine map_sfc_fine_grid_2_course_grid_vec( ml_sfc_course,sfc_nonods_course, &
                                                        ml_sfc_fine,sfc_nonods_fine)
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real, intent( out ) :: ml_sfc_course(sfc_nonods_course)
        real, intent( in ) :: ml_sfc_fine(sfc_nonods_fine)
! local variables...
        integer i_short, i_long
! 
        ml_sfc_course(:)=0.0
!        do i_short=1,sfc_nonods_course
!           i_long=(i_short-1)*2 + 1
!           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)

!           i_long=min( (i_short-1)*2 + 2, sfc_nonods_fine)----miss this out---
!           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
!        end do
! 
        do i_long=1,sfc_nonods_fine,2
           i_short=(i_long-1)/2 + 1
           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        end do
        do i_long=2,sfc_nonods_fine,2
           i_short=(i_long-1)/2 + 1
           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        end do
        
        return 
        end subroutine map_sfc_fine_grid_2_course_grid_vec
! 
! 
! 
! 
        subroutine map_sfc_course_grid_2_fine_grid_vec( ml_sfc_fine,sfc_nonods_fine, &
                                                        ml_sfc_course,sfc_nonods_course)
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real, intent( in ) :: ml_sfc_course(sfc_nonods_course)
        real, intent( out ) :: ml_sfc_fine(sfc_nonods_fine)
! local variables...
        integer i_short, i_long
! 
        do i_long=1,sfc_nonods_fine,2
           i_short=(i_long-1)/2 + 1
           ml_sfc_fine(i_long) = ml_sfc_course(i_short)
        end do
        do i_long=2,sfc_nonods_fine,2
           i_short=(i_long-1)/2 + 1
           ml_sfc_fine(i_long) = ml_sfc_course(i_short)
        end do
        
        return 
        end subroutine map_sfc_course_grid_2_fine_grid_vec
! 
! 
! 
! in python:
! t_new = time_step_filter_matrix( t_old, a_filter, fina,cola, ncola,nonods)
       subroutine time_step_filter_matrix(t_new, t_old, a_filter,   &
                                     fina,cola, ncola, nonods)
! this subroutine finds T^{n+1} = a_filter * T^n 
        implicit none
! nonods=number of finite element nodes in the mesh.
! dt = time step size. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
        integer, intent( in ) :: ncola, nonods
        real, intent( out ) :: t_new(nonods)
        real, intent( in ) :: a_filter(ncola)
        real, intent( in ) :: t_old(nonods)
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
! local variables...  
        integer count,nodi,icol
! 
        t_new=0.0
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              icol=cola(count)
              t_new(nodi) = t_new(nodi) + a_filter(count) * t_old(icol) 
           end do
        end do
! 
        return
        end subroutine time_step_filter_matrix
! 
! in python:
! a_filter = get_filter_matrix( a, ml, dt, fina,cola, ncola,nonods)
       subroutine get_filter_matrix(a_filter, a, ml, dt,  &
                                     fina,cola, ncola,nonods)
! this subroutine finds the matrix eqns a_filter = -M_L^{-1} ( -M_L + dt* A). 
! Time stepping can be realised with T^{n+1} = a_filter * T^n 
        implicit none
! nonods=number of finite element nodes in the mesh.
! dt = time step size. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
        integer, intent( in ) :: ncola, nonods
        real, intent( out ) :: a_filter(ncola)
        real, intent( in ) :: a(ncola), ml(nonods), dt
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
! local variables...  
        integer count,nodi,icol

        a_filter = dt*a 
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              icol=cola(count)
              if(icol==nodi) a_filter(count) = a_filter(count) - ml(nodi) 
           end do
        end do
! 
        do nodi=1,nonods
           do count=fina(nodi),fina(nodi+1)-1
              a_filter(count) = - a_filter(count)/ml(nodi) 
           end do
        end do
! 
        return
        end subroutine get_filter_matrix
! 
! 
! in python:
! a, b = u2r.get_fe_matrix_eqn(x_all, u, k, sig, s, fina,cola, ncola, ndglno, nonods,totele,nloc, ndim, ele_type)
! ml = the lumped mass 
        subroutine get_fe_matrix_eqn(a,b, ml, k, sig, s, u, x_all, &
                                     fina,cola, ndglno,  &
                                     ele_type, ndim, totele,nloc, ncola,nonods) 
! this subroutine finds the matrix eqns A T=b - that is it forms matrix a and vector b and the soln vector is T 
! although T is not needed here. 
        implicit none
! nonods=number of finite element nodes in the mesh.
! totele=no of elements in the mesh. 
! nloc=no of local nodes per element.
! ndim=no of dimensions.
! ele_type=element type index.
! 
! x_all contains the coordinates 
! and x_all(idim, (ele-1)*nloc+iloc) contains the idim'th coorindate for element number ele and local node 
! number iloc associated with element ele. 
! That is idim=1 corresponds to x coord, idim=2 y coordinate, idim=3 z coordinate.
!  
! u, k, sig, s the velocities, diffusion coefficient, absorption coefficient and source respectively of the differential equation. 
! The eqn we are solving is u\cdot\nabla T - \nabla \cdot k \nabla T + \sig T = s. 
! u(idim,inod) is the idim'th dimensional velocity (idim=1 corresponds to x-coord velocity) 
! of finite element node inod. 
! k(inod)=the diffusion coefficient of fem node nodi and similarly for sig, s. 
! 
! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
! fina(inod) start of the inod row of a matrix.
! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
!      1-----2-----3
!      !     !     !
!      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
!      4-----5-----6
! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
!           1  2  3  4  5  6 - column
! row 1    (X  X  0  X  X  0)
! row 2    (X  X  X  X  X  X)
! row 3    (0  X  X  0  X  X)
! row 4    (X  X  0  X  X  0)
! row 5    (X  X  X  X  X  X)
! row 6    (0  X  X  0  X  X)
! The comparact row storage only stores the non-zeros. 
! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
!                                                                                             fina(7)=29
! 
! 
! ndglno contains the gloabal node number given a local node number and element number. 
! Thus ndglno((ele-1)*nloc+iloc) contains the global node number for element number ele and local node 
! number iloc associated with element ele.
!         
! local variables...
! 2d:
!        integer, parameter :: totele=1,nloc=4,sngi=3, ngi=9, ndim=2, nface=4, max_face_list_no=2
! 3d:
!        integer, parameter :: totele=1,nloc=8,sngi=9, ngi=27, ndim=3, nface=6, max_face_list_no=4
!        real :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
!        real :: face_sn(sngi,nloc,nface), face_sn2(sngi,nloc,max_face_list_no), face_snlx(sngi,ndim,nloc,nface), face_sweigh(sngi,nface) 
        integer, intent( in ) :: ndim, totele, nloc, ele_type,  ncola, nonods
        real, intent( out ) :: a(ncola), b(nonods), ml(nonods)
        real, intent( in ) :: k(ndim,nonods), sig(nonods), s(nonods)
        real, intent( in ) :: u(ndim,nonods)
        real, intent( in ) :: x_all(ndim,nloc*totele)
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: ndglno(nloc*totele)
! local variables...
        real, allocatable :: n(:,:), nlx(:,:,:), weight(:)
        real, allocatable :: face_sn(:,:,:), face_sn2(:,:,:)
        real, allocatable :: face_snlx(:,:,:,:), face_sweigh(:,:) 
        real, allocatable :: nx(:,:,:),detwei(:),inv_jac(:,:,:)
        real, allocatable :: x_loc(:,:)
        real, allocatable :: ugi(:,:), kgi(:,:), sgi(:), siggi(:)
        real, allocatable :: l1(:), l2(:), l3(:), l4(:)  
        logical, allocatable :: lmark(:)
        integer npoly,ele, iloc,jloc,idim
        integer ifil,jfil,kfil, iiblock, jjblock, kkblock, i,j
        integer count, nodi, nodj
        integer sngi, ngi, nface, max_face_list_no, snloc
        real nxnx_k, nnx_u, nnsig, aa, max_x2
        logical d3, triangle_or_tet_high_order

! sngi=no of surface quadrature points of the faces - this is set to the max no of all faces.
! ngi=no of surface quadrature points of the faces. 
! nface=no of faces of each elemenet, nc no of fields to solve for.
    triangle_or_tet_high_order=.false.
    if( ((nloc==3).or.(nloc==6).or.(nloc==10).or.(nloc==15)) .and.(ndim==2)) then
       triangle_or_tet_high_order= .true. 
    endif
    if( ((nloc==4).or.(nloc==10).or.(nloc==20).or.(nloc==35)) .and.(ndim==3)) then
       triangle_or_tet_high_order= .true. 
    endif
! 
        if(ele_type.ge.100) then ! triangle or tet...
           if(ndim==2) then
              sngi=2; ngi=3; nface=3; max_face_list_no=2
              snloc=2
              if(triangle_or_tet_high_order) ngi=14
           else if(ndim==3) then
              sngi=3; ngi=4; nface=4; max_face_list_no=3
              snloc=3
              if(triangle_or_tet_high_order) ngi=45
           endif
        else ! rectangle...
           if(ndim==2) then
              sngi=3; ngi=9; nface=4; max_face_list_no=2
              snloc=2
           else if(ndim==3) then
              sngi=9; ngi=27; nface=6; max_face_list_no=4
              snloc=4
           endif
        endif

! 
        allocate( n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi) )
        allocate( face_sn(sngi,snloc,nface) )
        allocate( face_sn2(sngi,snloc,max_face_list_no) )
        allocate( face_snlx(sngi,ndim-1,snloc,nface), face_sweigh(sngi,nface) )

        allocate(nx(ngi,ndim,nloc),detwei(ngi),inv_jac(ngi,ndim,ndim))
!        allocate(nxnx(nloc,nloc),nnx(ndim,nloc,nloc))
        allocate(x_loc(ndim,nloc)) 

        allocate(ugi(ngi,ndim), kgi(ngi,ndim), sgi(ngi), siggi(ngi))

        npoly=1
!        ele_type=1

! form the shape functions...
      if(triangle_or_tet_high_order) then
        allocate(l1(ngi), l2(ngi), l3(ngi), l4(ngi) ) 
!        stop 22221
  !
        d3=(ndim==3)
!       print *,'****GOING bbbbbonkers 1'
        call TRIQUAold(L1, L2, L3, L4, WEIGHT, D3,NGI)

!        print *,'d3,nloc,ngi,ndim:',d3,nloc,ngi,ndim
!        print *,'l1,l2,l3:',l1,l2,l3
!        print *,'weight:',weight
!       print *,'****GOING bbbbbonkers 2'
        call SHATRInew(L1, L2, L3, L4, weight, &
          nloc,ngi,ndim,  n,nlx) 
!       print *,'****GOING bbbbbonkers 3'
!       stop 2821
      else
        print *,'going into get_shape_funs_with_faces'
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
      endif
! 
!        print *,'k:',k
!        print *,'sig:',sig
!        print *,'s:',s 
!        print *,'u:',u
!        print *,'x_all:',x_all
!        print *,'fina:',fina
!        print *,'cola:',cola 
!        print *,'ndglno:',ndglno
!        print *,'ele_type, ndim, totele,nloc, ncola,nonods:', &
!                 ele_type, ndim, totele,nloc, ncola,nonods
!        print *,'n:',n
!        print *,'nlx:',nlx
! obtain 
! for filters...
        a=0.0
        b=0.0 
        ml=0.0
        do ele = 1, totele ! VOLUME integral
             x_loc(:,:) = x_all(:,(ele-1)*nloc+1:ele*nloc)
             call det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )
!             print *,'nx, detwei, weight:',nx, detwei, weight
!
             ugi=0.0
             kgi=0.0
             sgi=0.0
             siggi=0.0
             do iloc=1,nloc
                nodi=ndglno((ele-1)*nloc+iloc) 
                do idim=1,ndim
                   ugi(:,idim) = ugi(:,idim) + n(:,iloc)*u(idim,nodi)
                   kgi(:,idim) = kgi(:,idim) + n(:,iloc)*k(idim,nodi)
                end do
                sgi(:) = sgi(:) + n(:,iloc)*s(nodi)
                siggi(:) = siggi(:) + n(:,iloc)*sig(nodi)
             end do

             do iloc=1,nloc
             nodi=ndglno((ele-1)*nloc+iloc) 
             b(nodi)=b(nodi)+sum( n(:,iloc)*detwei(:)*sgi(:) )
             ml(nodi)=ml(nodi)+sum( n(:,iloc)*detwei(:) )
             do jloc=1,nloc
                   nodj=ndglno((ele-1)*nloc+jloc) 
                   nxnx_k=0.0
                   nnx_u=0.0
                   do idim=1,ndim
                      nxnx_k = nxnx_k + sum( nx(:,idim,iloc)*kgi(:,idim)*nx(:,idim,jloc)*detwei(:) )
                      nnx_u = nnx_u + sum( n(:,iloc)*ugi(:,idim)*nx(:,idim,jloc)*detwei(:) )
                   end do
                   nnsig = sum( n(:,iloc)*siggi(:)*n(:,jloc)*detwei(:) )
                   aa=nxnx_k + nnx_u + nnsig
!                   do idim=1,1
!                      nxnx(iloc,jloc) = nxnx(iloc,jloc) + sum( nx(:,idim,iloc)*nx(:,idim,jloc)*detwei(:) )
!                   end do
!                   print *,'ugi:',ugi
!                   print *,'aa, nxnx_k, nnx_u, nnsig:',aa, nxnx_k, nnx_u, nnsig
                   do count=fina(nodi),fina(nodi+1)-1
                      if(cola(count)==nodj) a(count) = a(count) + aa
                   end do
             end do
             end do
        end do
! Boyang tweak for bc. ************
        if(.false.) then
        if(k(1,1)>0.5) then
          allocate( lmark(nonods) ) 
          lmark(:)=.false.
          max_x2=maxval(x_all(2,:))
          do nodi=1,nonods
             if(abs(x_all(2,nodi)-max_x2).lt.1.e-3) then
               lmark(nodi)=.true.
             endif
          end do
          if(.true.) then ! just take away off digonal terms...
            do nodi=1,nonods
               do count=fina(nodi),fina(nodi+1)-1
                  nodj=cola(count)
                  if(nodi.ne.nodj) then ! off diagonal terms...
                     if(lmark(nodi).or.lmark(nodj)) a(count)=0.0
                  endif
               end do
             end do
          else ! original method...
            do nodi=1,nonods
               if(abs(x_all(2,nodi)-max_x2).lt.1.e-3) then
                   do count=fina(nodi),fina(nodi+1)-1 ! change the diagonal...
                      if(cola(count)==nodi) a(count) = a(count)*4.0
                   end do
                endif
             end do
          endif
        endif
        endif
! Boyang tweak for bc. ************
! 
        return
        end subroutine get_fe_matrix_eqn
! 


!subroutine det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, jac )
   subroutine det_nlx( x_loc, n, nlx, nx, detwei, weight, ndim, nloc, ngi, INV_JAC )
  ! ****************************************************
  ! This sub form the derivatives of the shape functions
  ! ****************************************************
  ! x_loc: spatial nodes.
  ! n, nlx, nlx_lxx: shape function and local derivatives of the shape functions (nlx_lxx is local grad of the local laplacian)- defined by shape functional library.
  ! nx, nx_lxx: derivatives of the shape functions.
  ! detwei, inv_jac: determinant at the quadrature pots and inverse of Jacobian at quadrature pts.
  ! ndim,nloc,ngi: no of dimensions, no of local nodes within an element, no of quadrature points.
  ! nlx_nod, nx_nod: same as nlx and nx but formed at the nodes not quadrature points.
  implicit none
  integer, intent( in ) :: ndim,nloc,ngi

  REAL, DIMENSION( ndim,nloc ), intent( in ) :: x_loc
  REAL, DIMENSION( ngi, nloc ), intent( in ) :: N
  REAL, DIMENSION( ngi, ndim, nloc ), intent( in ) :: nlx
  REAL, DIMENSION( ngi ), intent( in ) :: WEIGHT
  REAL, DIMENSION( ngi ), intent( inout ) :: DETWEI
  REAL, DIMENSION( ngi, ndim, nloc ), intent( inout ) :: nx
  REAL, DIMENSION( ngi, ndim, ndim ), intent( inout ):: INV_JAC
  ! Local variables
  REAL :: AGI, BGI, CGI, DGI, EGI, FGI, GGI, HGI, KGI, A11, A12, A13, A21, &
          A22, A23, A31, A32, A33, DETJ
  INTEGER :: GI, L, IGLX, ii

  if (ndim==2) then
   ! conventional:
    do  GI=1,NGI! Was loop 331

      AGI=0.
      BGI=0.

      CGI=0.
      DGI=0.

      do  L=1,NLOC! Was loop 79

        AGI=AGI+NLX(GI,1,L)*x_loc(1,L)
        BGI=BGI+NLX(GI,1,L)*x_loc(2,L)

        CGI=CGI+NLX(GI,2,L)*x_loc(1,L)
        DGI=DGI+NLX(GI,2,L)*x_loc(2,L)

      end do ! Was loop 79

      DETJ= AGI*DGI-BGI*CGI
      DETWEI(GI)=ABS(DETJ)*WEIGHT(GI)
      ! For coefficient in the inverse mat of the jacobian.
      A11= DGI /DETJ
      A21=-BGI /DETJ

      A12=-CGI /DETJ
      A22= AGI /DETJ

      INV_JAC( GI, 1,1 )= A11
      INV_JAC( GI, 1,2 )= A21

      INV_JAC( GI, 2,1 )= A12
      INV_JAC( GI, 2,2 )= A22

      if(.false.) then
      do L=1,NLOC! Was loop 373
        NX(GI,1,L)= A11*NLX(GI,1,L)+A12*NLX(GI,2,L)
        NX(GI,2,L)= A21*NLX(GI,1,L)+A22*NLX(GI,2,L)
      end do ! Was loop 373
      else
          do L=1,NLOC
        NX(GI,1,L)= (DGI*NLX(GI,1,L) -BGI*NLX(GI,2,L))/DETJ
        NX(GI,2,L)= (-CGI*NLX(GI,1,L)+AGI*NLX(GI,2,L))/DETJ
!             NX(L,GI)=(DGI*NLX(L,GI)-BGI*NLY(L,GI))/DETJ
!             NY(L,GI)=(-CGI*NLX(L,GI)+AGI*NLY(L,GI))/DETJ
             !               NZ(L,GI)=0.0
          END DO
      endif
    end do ! GI Was loop 331

    !jac(1) = AGI; jac(2) = DGI ; jac(3) = BGI ; jac(4) = EGI

  elseif ( ndim.eq.3 ) then
    do  GI=1,NGI! Was loop 331

      AGI=0.
      BGI=0.
      CGI=0.

      DGI=0.
      EGI=0.
      FGI=0.

      GGI=0.
      HGI=0.
      KGI=0.

      do  L=1,NLOC! Was loop 79
        IGLX=L
        !ewrite(3,*)'xndgln, x, nl:', &
        !     iglx, l, x(iglx), y(iglx), z(iglx), NLX(L,GI), NLY(L,GI), NLZ(L,GI)
        ! NB R0 does not appear here although the z-coord might be Z+R0.
        AGI=AGI+NLX(GI,1,L)*x_loc(1,IGLX)
        BGI=BGI+NLX(GI,1,L)*x_loc(2,IGLX)
        CGI=CGI+NLX(GI,1,L)*x_loc(3,IGLX)

        DGI=DGI+NLX(GI,2,L)*x_loc(1,IGLX)
        EGI=EGI+NLX(GI,2,L)*x_loc(2,IGLX)
        FGI=FGI+NLX(GI,2,L)*x_loc(3,IGLX)

        GGI=GGI+NLX(GI,3,L)*x_loc(1,IGLX)
        HGI=HGI+NLX(GI,3,L)*x_loc(2,IGLX)
        KGI=KGI+NLX(GI,3,L)*x_loc(3,IGLX)
      end do ! Was loop 79

      DETJ=AGI*(EGI*KGI-FGI*HGI)&
          -BGI*(DGI*KGI-FGI*GGI)&
          +CGI*(DGI*HGI-EGI*GGI)
      DETWEI(GI)=ABS(DETJ)*WEIGHT(GI)
      ! ewrite(3,*)'gi, detj, weight(gi)', gi, detj, weight(gi)
      ! rsum = rsum + detj
      ! rsumabs = rsumabs + abs( detj )
      ! For coefficient in the inverse mat of the jacobian.
      A11= (EGI*KGI-FGI*HGI) /DETJ
      A21=-(DGI*KGI-FGI*GGI) /DETJ
      A31= (DGI*HGI-EGI*GGI) /DETJ

      A12=-(BGI*KGI-CGI*HGI) /DETJ
      A22= (AGI*KGI-CGI*GGI) /DETJ
      A32=-(AGI*HGI-BGI*GGI) /DETJ

      A13= (BGI*FGI-CGI*EGI) /DETJ
      A23=-(AGI*FGI-CGI*DGI) /DETJ
      A33= (AGI*EGI-BGI*DGI) /DETJ

      INV_JAC( GI, 1,1 )= A11
      INV_JAC( GI, 2,1 )= A21
      INV_JAC( GI, 3,1 )= A31
          !
      INV_JAC( GI, 1,2 )= A12
      INV_JAC( GI, 2,2 )= A22
      INV_JAC( GI, 3,2 )= A32
          !
      INV_JAC( GI, 1,3 )= A13
      INV_JAC( GI, 2,3 )= A23
      INV_JAC( GI, 3,3 )= A33

      do  L=1,NLOC! Was loop 373
        NX(GI,1,L)= A11*NLX(GI,1,L)+A12*NLX(GI,2,L)+A13*NLX(GI,3,L)
        NX(GI,2,L)= A21*NLX(GI,1,L)+A22*NLX(GI,2,L)+A23*NLX(GI,3,L)
        NX(GI,3,L)= A31*NLX(GI,1,L)+A32*NLX(GI,2,L)+A33*NLX(GI,3,L)
      end do ! Was loop 373
    end do ! GI Was loop 331
  end if
end subroutine det_nlx




  SUBROUTINE det_snlx_all( SNLOC, SNGI, SNDIM, ndim, XSL_ALL, SN, SNLX, &
                           SWEIGH, SDETWE, SAREA, NORMXN_ALL, NORMX_ALL )
!       inv_jac )
    IMPLICIT NONE

    INTEGER, intent( in ) :: SNLOC, SNGI, SNDIM, ndim
    REAL, DIMENSION( NDIM, SNLOC ), intent( in ) :: XSL_ALL
    REAL, DIMENSION( SNGI, SNLOC ), intent( in ) :: SN
    REAL, DIMENSION( SNGI, SNDIM, SNLOC ), intent( in ) :: SNLX
    REAL, DIMENSION( SNGI ), intent( in ) :: SWEIGH
    REAL, DIMENSION( SNGI ), intent( inout ) :: SDETWE
    REAL, intent( inout ) ::  SAREA
    REAL, DIMENSION( sngi, NDIM ), intent( inout ) :: NORMXN_ALL
    REAL, DIMENSION( NDIM ), intent( in ) :: NORMX_ALL
!    REAL, DIMENSION( NDIM,ndim ), intent( in ) :: inv_jac
    ! Local variables
    INTEGER :: GI, SL, IGLX
    REAL :: DXDLX, DXDLY, DYDLX, DYDLY, DZDLX, DZDLY
    REAL :: A, B, C, DETJ, RUB3, RUB4

    SAREA=0.
    if(ndim==3) then

       DO GI=1,SNGI

          DXDLX=0.
          DXDLY=0.
          DYDLX=0.
          DYDLY=0.
          DZDLX=0.
          DZDLY=0.

          DO SL=1,SNLOC
             DXDLX=DXDLX + SNLX(GI,1,SL)*XSL_ALL(1,SL)
             DXDLY=DXDLY + SNLX(GI,2,SL)*XSL_ALL(1,SL)
             DYDLX=DYDLX + SNLX(GI,1,SL)*XSL_ALL(2,SL)
             DYDLY=DYDLY + SNLX(GI,2,SL)*XSL_ALL(2,SL)
             DZDLX=DZDLX + SNLX(GI,1,SL)*XSL_ALL(3,SL)
             DZDLY=DZDLY + SNLX(GI,2,SL)*XSL_ALL(3,SL)
          END DO
          A = DYDLX*DZDLY - DYDLY*DZDLX
          B = DXDLX*DZDLY - DXDLY*DZDLX
          C = DXDLX*DYDLY - DXDLY*DYDLX

          DETJ=SQRT( A**2 + B**2 + C**2)
!          inv_jac(1,1)=DXDLX; inv_jac(1,2)=DXDLY; inv_jac(1,3)=DXDLZ
!          inv_jac(2,1)=DyDLX; inv_jac(2,2)=DyDLY; inv_jac(2,3)=DyDLZ
!          inv_jac(3,1)=DzDLX; inv_jac(3,2)=DzDLY; inv_jac(3,3)=DzDLZ
!          inv_jac=inv_jac/detj 
          SDETWE(GI)=DETJ*SWEIGH(GI)
          SAREA=SAREA+SDETWE(GI)

          ! Calculate the normal at the Gauss pts...
          ! Perform x-product. N=T1 x T2
          CALL NORMGI(NORMXN_ALL(GI,1),NORMXN_ALL(GI,2),NORMXN_ALL(GI,3), &
               DXDLX,DYDLX,DZDLX, DXDLY,DYDLY,DZDLY, &
               NORMX_ALL(1),NORMX_ALL(2),NORMX_ALL(3))
       END DO
    else ! 2D...
       DO GI=1,SNGI
          DXDLX=0.
          DXDLY=0.
          DYDLX=0.
          DYDLY=0.
          DZDLX=0.
          ! DZDLY=1 is to calculate the normal.
          DZDLY=1.
          DO SL=1,SNLOC
             DXDLX=DXDLX + SNLX(GI,1,SL)*XSL_ALL(1,SL)
             DYDLX=DYDLX + SNLX(GI,1,SL)*XSL_ALL(2,SL)
          END DO
          DETJ=SQRT( DXDLX**2 + DYDLX**2 )
          SDETWE(GI)=DETJ*SWEIGH(GI)
          SAREA=SAREA+SDETWE(GI)
          RUB3=0.0
          RUB4=0.0
          CALL NORMGI(NORMXN_ALL(GI,1),NORMXN_ALL(GI,2),RUB3, &
               DXDLX,DYDLX,DZDLX, DXDLY,DYDLY,DZDLY, &
               NORMX_ALL(1),NORMX_ALL(2),RUB4)
       END DO
!       print *,'XSL_ALL:',XSL_ALL
!       print *,'SNLX:',snlx
    endif

    RETURN

  END SUBROUTINE det_snlx_all



  SUBROUTINE NORMGI( NORMXN, NORMYN, NORMZN, &
       DXDLX, DYDLX, DZDLX, DXDLY, DYDLY, DZDLY, &
       NORMX, NORMY, NORMZ)
    ! Calculate the normal at the Gauss pts
    ! Perform x-product. N=T1 x T2
    implicit none
    REAL, intent( inout ) :: NORMXN, NORMYN, NORMZN
    REAL, intent( in )    :: DXDLX, DYDLX, DZDLX, DXDLY, DYDLY, DZDLY
    REAL, intent( in )    :: NORMX, NORMY, NORMZ
    ! Local variables
    REAL :: RN, SIRN

    CALL XPROD1( NORMXN, NORMYN, NORMZN, &
         DXDLX, DYDLX, DZDLX, &
         DXDLY, DYDLY, DZDLY )

    RN = SQRT( NORMXN**2 + NORMYN**2 + NORMZN**2 )

    SIRN = SIGN( 1.0 / RN, NORMXN * NORMX + NORMYN * NORMY + NORMZN * NORMZ )

    NORMXN = SIRN * NORMXN
    NORMYN = SIRN * NORMYN
    NORMZN = SIRN * NORMZN

    RETURN

  END SUBROUTINE NORMGI



  SUBROUTINE XPROD1( AX, AY, AZ, &
       BX, BY, BZ, &
       CX, CY, CZ )
    implicit none
    REAL, intent( inout ) :: AX, AY, AZ
    REAL, intent( in )    :: BX, BY, BZ, CX, CY, CZ

    ! Perform x-product. a=b x c
    AX =    BY * CZ - BZ * CY
    AY = -( BX * CZ - BZ * CX )
    AZ =    BX * CY - BY * CX

    RETURN
  END subroutine XPROD1




         subroutine get_shape_funs_spec(n, nlx, nlx_lxx, nlxx, weight, nlx_nod, &
          nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, face_sn, face_sn2, face_snlx, face_sweigh, &
          npoly, ele_type ) 
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
        integer, intent(in) :: nloc, snloc, sngi, ngi, ndim, nface, max_face_list_no
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives at the nodes. 
! face info:
! face_ele(iface, ele) = given the face no iface and element no return the element next to 
! the surface or if negative return the negative of the surface element number between element ele and face iface.
! face_list_no(iface, ele) returns the possible origantation number which defines the numbering 
! of the non-zeros of the nabouting element.  
        real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc), nlxx(ngi,nloc), nlx_lxx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: nlx_nod(nloc,ndim,nloc)
        real, intent(inout) :: face_sn(sngi,snloc,nface), face_sn2(sngi,snloc,max_face_list_no)
        real, intent(inout) :: face_snlx(sngi,ndim,snloc,nface), face_sweigh(sngi,nface) 
! npoly=order of polynomial in Cartesian space; ele_type=type of element including order of poly. 
        integer, intent(in) :: npoly,ele_type

! form the shape functions...
        call get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
          
! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod. 
        call get_high_order_shape_funs(n, nlx, nlx_lxx, nlxx, &
               weight, nlx_nod, nloc, ngi, ndim, &
               npoly, ele_type) 

        end subroutine get_shape_funs_spec




        subroutine get_high_order_shape_funs(n, nlx, nlx_lxx, nlxx, &
               weight, nlx_nod, nloc, ngi, ndim, &
               npoly, ele_type) 
! ********************************************************************
! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod. 
! ********************************************************************
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
        integer, intent(in) :: nloc, ngi, ndim, npoly, ele_type
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives in local coords.
! nlxx = the 2rd order local derivatives or Laplacian in local coords. 
        real, intent(in) :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: nlx_lxx(ngi,ndim,nloc),  nlxx(ngi,nloc)
        real, intent(inout) :: nlx_nod(nloc,ndim,nloc)
! local variables...
        integer iloc,jloc,idim
        real, allocatable :: nn(:,:), nn_inv(:,:), nnlx(:,:,:) 
        real, allocatable :: mat(:,:),mat2(:,:), x(:),b(:) 
        real, allocatable :: vec(:),sol(:),rhs_nod_lxx(:)
        
        allocate(nn(nloc,nloc), nn_inv(nloc,nloc), nnlx(nloc,nloc,ndim) )
        allocate(mat(nloc,nloc),mat2(nloc,nloc), x(nloc),b(nloc) )
        allocate(vec(nloc),sol(nloc),rhs_nod_lxx(nloc))

! Calculate high order derivatives of shape functions nlx_lxx, nlxx 
! and also calculate node-wise deriavtives of shape functions nlx_nod: 

! Calculate nodal values of nlx, that is nlx_nod:
! form nn
        do iloc=1,nloc
           do jloc=1,nloc
              nn(iloc,jloc)= sum( n(:,iloc)*n(:,jloc)*weight(:) )
              do idim=1,ndim
                 nnlx(iloc,jloc,idim)= sum( n(:,iloc)*nlx(:,idim,jloc)*weight(:) )
              end do
           end do
        end do
! find nn_inv as its easy to manipulate then although not efficient.
        nn_inv=nn
        call matinv(nn_inv,nloc,nloc,MAT,MAT2,X,B)

! Form nlx_nod
        do iloc=1,nloc
        do idim=1,ndim
           vec(:) = nnlx(iloc,:,idim)
           nlx_nod(iloc,idim,:) = matmul( nn_inv(:,:), vec(:) )
        end do
        end do

! Form nlxx from nlx_nod...
        nlxx=0.0
        do iloc=1,nloc
           do idim=1,ndim
              vec(:) = sum( nnlx(iloc,:,idim)*nlx_nod(iloc,idim,:) )
              sol(:) = matmul( nn_inv(:,:), vec(:) )
              nlxx(iloc,:) = nlxx(iloc,:) +sol(:) 
           end do
        end do

! Form nlx_lxx: 
        do iloc=1,nloc
           rhs_nod_lxx(:)=nlxx(iloc,:)
           do idim=1,ndim
              nlx_lxx(:,idim,iloc) = nlx(:,idim,iloc)*rhs_nod_lxx(:) 
           end do
        end do
          
        end subroutine get_high_order_shape_funs

     



        subroutine get_shape_funs_with_faces(n, nlx, weight,  &
               nloc, snloc, sngi, ngi, ndim, nface,max_face_list_no, &
               face_sn, face_sn2, face_snlx, face_sweigh, &
               npoly,ele_type) 
        implicit none 
! nloc=no of local nodes per element
! ngi=no of quadrature points 
! ndim=no of dimensions. 
! ele_type= element type
        integer, intent(in) :: nloc, snloc, sngi, ngi, ndim, nface, max_face_list_no
! shape functions....
! if .not.got_shape_funs then get the shape functions else assume we have them already
! n, nlx are the volume shape functions and their derivatives in local coordinates.
! weight are the weights of the quadrature points.
! nlx_nod are the derivatives of the local coordinates at the nods. 
! nlx_lxx = the 3rd order local derivatives at the nodes. 
! face info:
! face_ele(iface, ele) = given the face no iface and element no return the element next to 
! the surface or if negative return the negative of the surface element number between element ele and face iface.
! face_list_no(iface, ele) returns the possible origantation number which defines the numbering 
! of the non-zeros of the nabouting element.  
        real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc), weight(ngi)
        real, intent(inout) :: face_sn(sngi,snloc,nface)
        real, intent(inout) :: face_sn2(sngi,snloc,max_face_list_no)
        real, intent(inout) :: face_snlx(sngi,ndim-1,snloc,nface)
        real, intent(inout) :: face_sweigh(sngi,nface) 
! npoly=order of polynomial in Cartesian space; ele_type=type of element including order of poly. 
        integer, intent(in) :: npoly, ele_type
! local variables...
        integer is_triangle_or_tet
        parameter(is_triangle_or_tet=100) 
        integer sndim, suf_ngi, suf_ndim, suf_nloc, ipoly, IQADRA, iface
        integer idim,siloc
        logical with_time_slab
        real, allocatable :: rdum1(:), rdum2(:), rdum3(:) 
        real, allocatable :: sn(:,:),sn2(:,:),snlx(:,:,:),sweight(:) 
        real, allocatable :: suf_n(:,:),suf_nlx(:,:,:),suf_weight(:)

! allocate memory...
        print *,'just about to allocate'
        print *,'nloc, sngi, ngi, ndim, nface,max_face_list_no:', &
                 nloc, sngi, ngi, ndim, nface,max_face_list_no
        print *,'npoly,ele_type:',npoly,ele_type
        allocate(rdum1(10000), rdum2(10000), rdum3(10000) )
        print *,'here 1'
        allocate(sn(sngi,snloc),sn2(sngi,snloc),snlx(sngi,ndim-1,snloc),sweight(sngi) )
        print *,'here 2'
        allocate(suf_n(sngi,snloc),suf_nlx(sngi,ndim-1,snloc),suf_weight(sngi) )
        print *,'here 3'

        ipoly=npoly
!        IQADRA=IPOLY+1
        IQADRA=1

        sndim=ndim-1
        if(ele_type < is_triangle_or_tet) then ! not triangle...
! 
           with_time_slab=.true.
    if( ((nloc==3).or.(nloc==6).or.(nloc==10).or.(nloc==15)) .and.(ndim==2)) then
       with_time_slab= .false. 
    endif
    if( ((nloc==4).or.(nloc==10).or.(nloc==20).or.(nloc==35)) .and.(ndim==3)) then
       with_time_slab= .false. 
    endif

           print *,'get_shape_funs'
           call get_shape_funs(ngi,nloc,ndim,  &
             weight,n,nlx, ipoly,iqadra, &
             sngi, snloc, sndim, sweight,sn,snlx, with_time_slab   )
           print *,'finished get_shape_funs'
           print *,'sn:',sn
           print *,'snlx:',snlx
           print *,'sweight:',sweight
                 do siloc=1,snloc 
                    do idim=1,ndim-1
       print *,'snlx(:,idim,siloc):',idim,siloc,snlx(:,idim,siloc)
                    end do
                 end do
           do iface=1,nface
             face_sn(:,:,iface) = sn(:,:) 
             face_snlx(:,:,:,iface) = snlx(:,:,:) 
             face_sweigh(:,iface) = sweight(:) 
           end do
           face_sn2(:,:,1:max_face_list_no) = face_sn(:,:,1:max_face_list_no) 

        else 
!           allocate(rdum1(10000),rdum2(10000),rdum3(10000))
! gives a surface triangle with time slab in surface integral. 
           print *,'going into get_shape_funs ***is a triangle or tet'
           call get_shape_funs(ngi,nloc,ndim,  &
              weight,n,nlx, ipoly,iqadra, &
              sngi, snloc, sndim, sweight,sn,snlx, .false.   )
           print *,'out of get_shape_funs'
! return a surface tet...
           suf_ngi=NGI/IQADRA
           suf_ndim=ndim-1
           suf_nloc=NLOC/(IPOLY+1)
           call get_shape_funs(suf_ngi,suf_nloc,suf_ndim,  &
              suf_weight,suf_n,suf_nlx, ipoly,iqadra, &
              sngi, snloc, sndim, rdum1,rdum2,rdum3, .false.   )

        endif

        end subroutine get_shape_funs_with_faces
     



       subroutine get_shape_funs(ngi,nloc,ndim,  &
             weight,n,nlx, ipoly,iqadra, &
             sngi, snloc, sndim, sweight,sn,snlx, with_time_slab   )
! ***************************************
! form volume and surface shape functions
! ***************************************
    IMPLICIT NONE
    INTEGER, intent(in) :: sngi, NGI, NLOC, snloc, ndim, sndim
    INTEGER, intent(in) :: IPOLY,IQADRA
    logical, intent(in) :: with_time_slab
    REAL, intent(inout) ::  n(ngi,nloc) 
    REAL, intent(inout) ::  nlx(ngi,ndim,nloc)
    REAL, intent(inout) ::  sn(sngi, snloc) 
    REAL, intent(inout) ::  snlx(sngi, sndim, snloc)  
    real, intent(inout) :: WEIGHT(ngi), sWEIGHT(sngi)
! local variables...
    integer mloc, NDNOD, snloc_temp, idim,siloc
    logical square
    real, allocatable :: m(:)

    allocate(m(10000)) 
    mloc=1
    SQUARE = .true. 
    if( ((nloc==3).or.(nloc==6).or.(nloc==10).or.(nloc==15)) .and.(ndim==2)) then
       SQUARE = .false. 
    endif
    if( ((nloc==4).or.(nloc==10).or.(nloc==20).or.(nloc==35)) .and.(ndim==3)) then
       SQUARE = .false. 
    endif
     print *,'++++++square=',square

    IF(SQUARE) THEN ! Square in up to 4D
! volumes
!       print *,'going into interface_SPECTR NGI,NLOC,ndim:',NGI,NLOC,ndim
       call interface_SPECTR(NGI,NLOC,WEIGHT,N,NLX, ndim, IPOLY,IQADRA  )
!       print *,'out of interface_SPECTR'
! surfaces...
       NDNOD =INT((NLOC**(1./real(ndim) ))+0.1)
       snloc_temp=NDNOD**(ndim-1) 
       sWEIGHT=0.0; sN=0.0; sNLX=0.0
!       print *,'surfaces going into interface_SPECTR'
       call interface_SPECTR(sNGI,sNLOC_temp,sWEIGHT,sN,sNLX, ndim-1, IPOLY,IQADRA  )
!       print *,'surfaces out of interface_SPECTR'
        print *,'sNGI,sNLOC_temp,ndim,IPOLY,IQADRA:', &
                 sNGI,sNLOC_temp,ndim,IPOLY,IQADRA
                 do siloc=1,snloc 
                    do idim=1,sndim
       print *,'snlx(:,idim,siloc):',idim,siloc,snlx(:,idim,siloc)
                    end do
                 end do
        print *,'sWEIGHT:',sWEIGHT
        print *,'sndim:',sndim
       
    ELSE
! volume tet plus time slab...
!       print *,'triangles or tets ipoly,iqadra, with_time_slab:',ipoly,iqadra, with_time_slab
!       if(.true.) then
!       else
       call triangles_tets_with_time( nloc,ngi, ndim, &
               n,nlx, weight, ipoly,iqadra, with_time_slab) 
! surfaces...
!       print *,'surfaces -tets or triangles...'
       call triangles_tets_with_time( snloc,sngi, ndim-1, &
              sn,snlx, sweight, ipoly,iqadra, with_time_slab) 
!       endif 
    ENDIF
     return
     end subroutine get_shape_funs




     subroutine triangles_tets_with_time( nloc,ngi, ndim, &
                  n,nlx, weight, ipoly,iqadra, with_time_slab) 
! ****************************************************************************************
     implicit none
     integer, intent(in) :: nloc,ngi,ndim, ipoly,iqadra
     real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc) 
     real, intent(inout) :: weight(ngi)
     logical, intent(in) :: with_time_slab
! local variables...
     integer nloc_space, ngi_space, ndim_space,   nloc_t, ngi_t, ndim_t
     logical d3
     real, allocatable :: l1(:), l2(:), l3(:), l4(:)
     real, allocatable :: weight_space(:), n_space(:,:), nlx_space(:,:,:)
     real, allocatable :: weight_t(:), n_t(:,:), nlx_t(:,:,:)

     if(with_time_slab) then ! form a space-time slab.

        nloc_t=nloc/(ipoly+1)
        NGI_T=IQADRA
!        ngi_t=ngi/(ipoly+2)
        ndim_t = 1

        nloc_space=nloc/nloc_t 
        ngi_space=ngi/ngi_t
        ndim_space = ndim-1
        allocate(l1(ngi_space), l2(ngi_space), l3(ngi_space), l4(ngi_space) )
        allocate(weight_space(ngi_space), n_space(ngi_space,nloc_space), &
                 nlx_space(ngi_space,ndim_space,nloc_space) )

        allocate(weight_t(ngi_t), n_t(ngi_t,nloc_t), nlx_t(ngi_t,ndim_t,nloc_t) )

! triangles or tetrahedra...
        call SHATRInew(L1, L2, L3, L4, WEIGHT_space, &
          NLOC_space,NGI_space,ndim_space,  n_space,nlx_space)

! extend into time domain...
        call interface_SPECTR(NGI_t,NLOC_t,WEIGHT_t,N_t,NLX_t, &
                              ndim_t, IPOLY,IQADRA  )

! combine-space time...
        call make_space_time_shape_funs(nloc_space,ngi_space, ndim_space, &
                                     n_space,nlx_space, weight_space, &
                                     nloc_t,ngi_t, ndim_t, n_t, nlx_t, weight_t, &
                                     nloc,ngi, ndim, n, nlx, weight ) 
     else ! just a triangle or tet without time slab...
! triangles or tetrahedra...
        allocate(l1(ngi), l2(ngi), l3(ngi), l4(ngi) ) 
!        stop 22221
  !
        d3=(ndim==3)
        call TRIQUAold(L1, L2, L3, L4, WEIGHT, D3,NGI)

!        print *,'d3,nloc,ngi,ndim:',d3,nloc,ngi,ndim
!        print *,'l1,l2,l3:',l1,l2,l3
!        print *,'weight:',weight
        call SHATRInew(L1, L2, L3, L4, weight, &
          nloc,ngi,ndim,  n,nlx)
!        print *,'n:',n
!        stop 282
     endif

     end subroutine triangles_tets_with_time
  ! 



     subroutine make_space_time_shape_funs(nloc_space,ngi_space, ndim_space, &
                                           n_space,nlx_space, weight_space, &
                                           nloc_t,ngi_t, ndim_t, n_t, nlx_t, weight_t, &
                                           nloc,ngi, ndim, n, nlx, weight ) 
! ****************************************************************************************
! this sub convolves the space and time shape functions to get space-time shape functions. 
! these are returned in n, nlx where nloc=no of local nodes and ngi =no of quadrature pts.
! ****************************************************************************************
     implicit none
     integer, intent(in) :: nloc_space,nloc_t, ngi_space,ngi_t, nloc,ngi, &
                            ndim_space,ndim_t, ndim
     real, intent(in) :: n_space(ngi_space,nloc_space),n_t(ngi_t,nloc_t)
     real, intent(in) :: nlx_space(ngi_space,ndim_space,nloc_space), &
                         nlx_t(ngi_t,ndim_t,nloc_t)
     real, intent(in) :: weight_space(ngi_space), weight_t(ngi_t)
     real, intent(inout) :: n(ngi,nloc), nlx(ngi,ndim,nloc) 
     real, intent(inout) :: weight(ngi)
! local...
     integer iloc_space, iloc_t, iloc, gi_space, gi_t, gi, &
             idim_space, idim_t, idim 
     
! remember sngi>=the needed sngi for a surface element.
! similarly for nloc
     n=0.0
     nlx=0.0 
     weight=0.0
     do iloc_space=1,nloc_space
        do iloc_t=1,nloc_t
           iloc=(iloc_t-1)*nloc_space + iloc_space
           do gi_space=1,ngi_space
              do gi_t=1,ngi_t
                 gi=(gi_t-1)*ngi_space + gi_space
                 n(gi,iloc) = n_space(gi_space,iloc_space)*n_t(gi_t,iloc_t) 
                 do idim_space=1,ndim_space
                    nlx(gi,idim_space,iloc) = nlx_space(gi_space,idim_space,iloc_space)*n_t(gi_t,iloc_t)
                 end do
                 idim = ndim_space+1
                 nlx(gi,idim,iloc) = n(gi_space,iloc_space)*nlx_t(gi_t,1,iloc_t)
     end do; end do; end do; end do
! the weights...
     do gi_space=1,ngi_space
        do gi_t=1,ngi_t
           gi=(gi_t-1)*ngi_space + gi_space
           weight(gi) = weight_space(gi_space)*weight_t(gi_t)
        end do
     end do
     end subroutine make_space_time_shape_funs
  ! 


  ! 
       subroutine interface_SPECTR(NGI,NLOC,WEIGHT,N,NLX, ndim, IPOLY,IQADRA  )
    IMPLICIT NONE
    INTEGER , intent(in) :: NGI, NLOC, ndim
    INTEGER , intent(in) :: IPOLY,IQADRA
    REAL, dimension(ngi,nloc), intent(inout) ::  N
    REAL, dimension(ngi,ndim,nloc), intent(inout) ::  NLX   
    real, dimension(ngi), intent(inout) :: WEIGHT
! local variables...
    logical d2,d3,d4
    integer mloc,idim
    real, allocatable :: m(:),NLX_TEMP(:,:,:)

    allocate(m(10000),NLX_TEMP(NGI,4,nloc)) 
    mloc=nloc ! 1
    d2=.false.; d3=.false.; d4=.false.
    if(ndim==2) d2=.true.
    if(ndim==3) d3=.true.
    if(ndim==4) d4=.true.

!    IF(SQUARE) THEN ! Square in up to 4D
! volumes
       print *,'going into SPECTR'
       call SPECTR(NGI,NLOC,MLOC, &
       &      M,WEIGHT,N,NLX_temp(:,1,:),NLX_temp(:,2,:), &
              NLX_temp(:,3,:),NLX_temp(:,4,:),D4,D3,D2, IPOLY,IQADRA  )
       print *,'out of SPECTR'

    do idim=1,ndim
       nlx(:,idim,:) = nlx_temp(:,idim,:) 
    end do

    end subroutine interface_SPECTR




  SUBROUTINE SPECTR(NGI,NLOC,MLOC, &
              M,WEIGHT,N,NLX, & 
              NLY,NLZ,NLT,D4,D3,D2, IPOLY,IQADRA  )
    IMPLICIT NONE
    INTEGER, intent(in) :: NGI, NLOC, MLOC
    INTEGER, intent(in) :: IPOLY,IQADRA
    REAL, dimension(ngi,nloc), intent(inout) ::  M, N, NLX, NLY, NLZ, NLT
    real, dimension(ngi), intent(inout) :: WEIGHT
    logical, intent(in) :: d2,d3,d4
    !Local variables
    REAL :: RGPTWE
    REAL , dimension (300) :: WEIT,NODPOS,QUAPOS
    INTEGER :: GPOI
    LOGICAL :: DIFF,NDIFF
    INTEGER :: NDGI,NDNOD,NMDNOD,IGR,IGQ,IGP,KNOD,JNOD,INOD,ILOC,lnod,igs
    REAL :: LXGP,LYGP,LZGP,LTGP
! SPECFU is a real function
    real :: SPECFU
    ! This subroutine defines a spectal element.
    ! IPOLY defines the element type and IQADRA the quadrature.
    ! In 2-D the spectral local node numbering is as..
    ! 7 8 9
    ! 4 5 6
    ! 1 2 3
    ! For 3-D...
    ! lz=-1
    ! 3 4
    ! 1 2
    ! and for lz=1
    ! 7 8
    ! 5 6

    !ewrite(3,*)'inside SPECTR IPOLY,IQADRA', IPOLY,IQADRA
    !
    DIFF=.TRUE.
    NDIFF=.FALSE.
    IF(D4) THEN
       NDGI  =INT((NGI**(1./4.))+0.1)
       NDNOD =INT((NLOC**(1./4.))+0.1)
       NMDNOD=INT((MLOC**(1./4.))+0.1)
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       !ewrite(3,*)'outside GTROOT'
       do  IGS=1,NDGI! Was loop 101
       do  IGR=1,NDGI! Was loop 101
          do  IGQ=1,NDGI! Was loop 101
             do  IGP=1,NDGI! Was loop 101
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI + (IGS-1)*NDGI*NDGI*NDGI
                !
                !           WEIGHT(GPOI)
                !     &        =RGPTWE(IGP,NDGI,.TRUE.)*RGPTWE(IGQ,NDGI,.TRUE.)
                !     &        *RGPTWE(IGR,NDGI,.TRUE.)
                WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)*WEIT(IGR)*WEIT(IGS)
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                LTGP=QUAPOS(IGS)
                ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
                ! else return the Gauss-pt.
                !
                do  LNOD=1,NDNOD! Was loop 20
                do  KNOD=1,NDNOD! Was loop 20
                   do  JNOD=1,NDNOD! Was loop 20
                      do  INOD=1,NDNOD! Was loop 20
                         ILOC=INOD + (JNOD-1)*NDNOD + (KNOD-1)*NDNOD*NDNOD + &
                             (LNOD-1)*NDNOD*NDNOD*NDNOD
                         !
                         N(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLX(GPOI,ILOC) = &
                              SPECFU(DIFF,LXGP, INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLY(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LTGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLZ(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LTGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLT(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LtGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 20
                   end do ! Was loop 20
                end do ! Was loop 20
                end do ! Was loop 20
             end do ! Was loop 101
          end do ! Was loop 101
       end do ! Was loop 101
       end do ! Was loop 101
       !
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'2about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       !ewrite(3,*)'2out of GTROOT'

       do  IGS=1,NDGI! Was loop 102
       do  IGR=1,NDGI! Was loop 102
          do  IGQ=1,NDGI! Was loop 102
             do  IGP=1,NDGI! Was loop 102
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI &
                    + (IGS-1)*NDGI*NDGI*NDGI
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                LTGP=QUAPOS(IGS)

                do  LNOD=1,NMDNOD! Was loop 30
                do  KNOD=1,NMDNOD! Was loop 30
                   do  JNOD=1,NMDNOD! Was loop 30
                      do  INOD=1,NMDNOD! Was loop 30
                         ILOC=INOD + (JNOD-1)*NMDNOD + (KNOD-1)*NMDNOD*NMDNOD &
                             + (LNOD-1)*NMDNOD*NMDNOD*NMDNOD
                         !
                         M(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)  &
                              *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS) &
                              *SPECFU(NDIFF,LZGP,KNOD,NMDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LtGP,LNOD,NMDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 30
                   end do ! Was loop 30
                end do ! Was loop 30
                end do ! Was loop 30
             end do ! Was loop 102
          end do ! Was loop 102
       end do ! Was loop 102
       end do ! Was loop 102
    ENDIF ! ENDIF IF(D4) THEN

    IF(D3) THEN
       NDGI  =INT((NGI**(1./3.))+0.1)
       NDNOD =INT((NLOC**(1./3.))+0.1)
       NMDNOD=INT((MLOC**(1./3.))+0.1)
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       print *,'volume 3d before GTROOT'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       print *,'volume 3d after GTROOT'
       !ewrite(3,*)'outside GTROOT'
       do  IGR=1,NDGI! Was loop 101
          do  IGQ=1,NDGI! Was loop 101
             do  IGP=1,NDGI! Was loop 101
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI
                !
                !           WEIGHT(GPOI)
                !     &        =RGPTWE(IGP,NDGI,.TRUE.)*RGPTWE(IGQ,NDGI,.TRUE.)
                !     &        *RGPTWE(IGR,NDGI,.TRUE.)
                WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)*WEIT(IGR)
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)
                ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
                ! else return the Gauss-pt.
                !
                do  KNOD=1,NDNOD! Was loop 20
                   do  JNOD=1,NDNOD! Was loop 20
                      do  INOD=1,NDNOD! Was loop 20
                         ILOC=INOD + (JNOD-1)*NDNOD + (KNOD-1)*NDNOD*NDNOD
                         !
                         N(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
!            print *,'GPOI,iloc,N(GPOI,ILOC):',GPOI,iloc,N(GPOI,ILOC)
                         !
                         NLX(GPOI,ILOC) = &
                              SPECFU(DIFF,LXGP, INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
!            print *,'GPOI,iloc,NLX(GPOI,ILOC):',GPOI,iloc,NLX(GPOI,ILOC)
                         !
                         NLY(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LZGP,KNOD,NDNOD,IPOLY,NODPOS)
                         !
                         NLZ(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)&
                              *SPECFU(DIFF,LZGP, KNOD,NDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 20
                   end do ! Was loop 20
                end do ! Was loop 20
             end do ! Was loop 101
          end do ! Was loop 101
       end do ! Was loop 101
       !
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'2about to go into inside GTROOT IPOLY,IQADRA',IPOLY,IQADRA
       print *,'surface 3d before GTROOT'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       print *,'surface 3d after GTROOT'
       !ewrite(3,*)'2out of GTROOT'
       do  IGR=1,NDGI! Was loop 102
          do  IGQ=1,NDGI! Was loop 102
             do  IGP=1,NDGI! Was loop 102
                GPOI=IGP + (IGQ-1)*NDGI + (IGR-1)*NDGI*NDGI
                !
                LXGP=QUAPOS(IGP)
                LYGP=QUAPOS(IGQ)
                LZGP=QUAPOS(IGR)

                do  KNOD=1,NMDNOD! Was loop 30
                   do  JNOD=1,NMDNOD! Was loop 30
                      do  INOD=1,NMDNOD! Was loop 30
                         ILOC=INOD + (JNOD-1)*NMDNOD + (KNOD-1)*NMDNOD*NMDNOD
                         !
                         M(GPOI,ILOC) = &
                              SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)  &
                              *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS) &
                              *SPECFU(NDIFF,LZGP,KNOD,NMDNOD,IPOLY,NODPOS)
                         !
                      end do ! Was loop 30
                   end do ! Was loop 30
                end do ! Was loop 30
             end do ! Was loop 102
          end do ! Was loop 102
       end do ! Was loop 102
       print *,'still 3d'
    ENDIF
    !
    IF(D2) THEN
!       return
       print *,'here 1.1'
       NDGI  =INT((NGI**(1./2.))+0.1)
       NDNOD =INT((NLOC**(1./2.))+0.1)
       NMDNOD=INT((MLOC**(1./2.))+0.1)
       print *,'ndgi,ndnod,nmdnod:',ndgi,ndnod,nmdnod
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       print *,'here 1.2'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       print *,'here 1.3'
!       return
       do  IGQ=1,NDGI! Was loop 10
          do  IGP=1,NDGI! Was loop 10
             GPOI=IGP + (IGQ-1)*NDGI
             !
             WEIGHT(GPOI)=WEIT(IGP)*WEIT(IGQ)
             !
             LXGP=QUAPOS(IGP)
             LYGP=QUAPOS(IGQ)
             ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
             ! else return the Gauss-pt.
             !
             do  JNOD=1,NDNOD! Was loop 120
                do  INOD=1,NDNOD! Was loop 120
                   ILOC=INOD + (JNOD-1)*NDNOD
                   !
                   N(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                   NLX(GPOI,ILOC) = &
                        SPECFU(DIFF, LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                   NLY(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS) &
                        *SPECFU(DIFF, LYGP,JNOD,NDNOD,IPOLY,NODPOS)
                   !
                end do ! Was loop 120
             end do ! Was loop 120
          end do ! Was loop 10
       end do ! Was loop 10
       print *,'here 1.4'
!       return
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       print *,'here 1.5'
       do  IGQ=1,NDGI! Was loop 11
          do  IGP=1,NDGI! Was loop 11
             GPOI=IGP + (IGQ-1)*NDGI
             LXGP=QUAPOS(IGP)
             LYGP=QUAPOS(IGQ)
             do  JNOD=1,NMDNOD! Was loop 130
                do  INOD=1,NMDNOD! Was loop 130
                   ILOC=INOD + (JNOD-1)*NMDNOD
                   !
                   M(GPOI,ILOC) = &
                        SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS) &
                        *SPECFU(NDIFF,LYGP,JNOD,NMDNOD,IPOLY,NODPOS)
                   !
                end do ! Was loop 130
             end do ! Was loop 130
             !
          end do ! Was loop 11
       end do ! Was loop 11
       print *,'here 1.6 d2=',d2
    ENDIF
    !
    !
    IF((.NOT.D2).AND.(.NOT.D3).and.(.not.D4)) THEN
       NDGI  =NGI
       NDNOD =NLOC
       NMDNOD=MLOC
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
       !ewrite(3,*)'NDGI,NDNOD,NLOC:',NDGI,NDNOD,NLOC
       !ewrite(3,*)'WEIT(1:ndgi):',WEIT(1:ndgi)
       !ewrite(3,*)'NODPOS(1:ndnod):',NODPOS(1:ndnod)
       !ewrite(3,*)'QUAPOS(1:ndgi):',QUAPOS(1:ndgi)
       do  IGP=1,NDGI! Was loop 1000
          GPOI=IGP
          !
          WEIGHT(GPOI)=WEIT(IGP)
          !
          LXGP=QUAPOS(IGP)
          ! NB If TRUE in function RGPTWE then return the Gauss-pt weight
          ! else return the Gauss-pt.
          !
          do  INOD=1,NDNOD! Was loop 12000
             ILOC=INOD
             !
             N(GPOI,ILOC) = &
                  SPECFU(NDIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)
             !
             NLX(GPOI,ILOC) = &
                  SPECFU(DIFF, LXGP,INOD,NDNOD,IPOLY,NODPOS)
             !ewrite(3,*)'ILOC,GPOI,N(GPOI,ILOC),NLX(GPOI,ILOC):', &
             !         ILOC,GPOI,N(GPOI,ILOC),NLX(GPOI,ILOC)
             !
          end do ! Was loop 12000
       end do ! Was loop 1000
       !ewrite(3,*)'n WEIGHT:',WEIGHT
       !
       ! Find the roots of the quadrature points and nodes
       ! also get the weights.
       !ewrite(3,*)'this is for m which we dont care about:'
       CALL GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NMDNOD)
       do  IGP=1,NDGI! Was loop 1100
          GPOI=IGP
          LXGP=QUAPOS(IGP)
          do  INOD=1,NMDNOD! Was loop 13000
             ILOC=INOD
             !
             M(GPOI,ILOC) = &
                  SPECFU(NDIFF,LXGP,INOD,NMDNOD,IPOLY,NODPOS)
             !
          end do ! Was loop 13000
          !
       end do ! Was loop 1100
       !ewrite(3,*)'...finished this is for m which we dont care about:'
    ENDIF
    print *,'just leaving spectr'
  END SUBROUTINE SPECTR




  SUBROUTINE GTROOT(IPOLY,IQADRA,WEIT,NODPOS,QUAPOS,NDGI,NDNOD)
    IMPLICIT NONE
    INTEGER , intent(in) :: IPOLY,IQADRA,NDGI,NDNOD
    REAL , dimension(NDNOD), intent(inout) :: NODPOS
    REAL , dimension(NDGI), intent(inout) :: WEIT,QUAPOS
    !Local variables
    LOGICAL :: GETNDP
    !     This sub returns the weights WEIT the quadrature points QUAPOS and
    !     the node points NODPOS.
    !     NODAL POISTIONS ******
    !     NB if GETNDP then find the nodal positions
!    print *,'IQADRA=',IQADRA
!    stop 38
    GETNDP=.TRUE.
    !     Compute standard Lagrange nodal points
    IF(IQADRA.EQ.1) CALL LAGROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !     Compute Chebyshev-Gauss-Lobatto nodal points.
    IF(IQADRA.EQ.2) CALL CHEROT(NODPOS,NODPOS,NDNOD,GETNDP)
    IF(IQADRA.EQ.3) CALL CHEROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !     Compute Legendre-Gauss-Lobatto nodal points.
    IF(IQADRA.EQ.4) CALL LEGROT(NODPOS,NODPOS,NDNOD,GETNDP)
    !
    !     QUADRATURE************
    GETNDP=.FALSE.
    !     Compute standard Gauss quadrature. weits and points
    IF(IQADRA.EQ.1) CALL LAGROT(WEIT,QUAPOS,NDGI,GETNDP)
    !     Compute Chebyshev-Gauss-Lobatto quadrature.
    IF(IQADRA.EQ.2) CALL CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    IF(IQADRA.EQ.3) CALL CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    !     Compute Legendre-Gauss-Lobatto quadrature.
    IF(IQADRA.EQ.4) CALL LEGROT(WEIT,QUAPOS,NDGI,GETNDP)
  END SUBROUTINE GTROOT




  REAL FUNCTION SPECFU(DIFF,LXGP,INOD,NDNOD,IPOLY,NODPOS)
    LOGICAL , intent(in):: DIFF
    INTEGER , intent(in) :: INOD,NDNOD,IPOLY
    REAL , intent(inout) :: LXGP
    real, dimension(NDNOD), intent(inout) :: NODPOS
    !     INOD contains the node at which the polynomial is associated with
    !     LXGP is the position at which the polynomial is to be avaluated.\
    !     If(DIFF) then find the D poly/DX.
! local variables (functions)
    real LAGRAN,CHEBY1,CHEBY2,LEGEND
    !
    IF(IPOLY.EQ.1) SPECFU=LAGRAN(DIFF,LXGP,INOD,NDNOD,NODPOS)
!         print *,'ipoly,NODPOS:',ipoly,NODPOS
!          stop 2921
    !
    IF(IPOLY.EQ.2) SPECFU=CHEBY1(DIFF,LXGP,INOD,NDNOD,NODPOS)
    !
    IF(IPOLY.EQ.3) SPECFU=CHEBY2(DIFF,LXGP,INOD,NDNOD,NODPOS)
    !
    IF(IPOLY.EQ.4) SPECFU=LEGEND(DIFF,LXGP,INOD,NDNOD,NODPOS)
  END FUNCTION SPECFU



  SUBROUTINE CHEROT(WEIT,QUAPOS,NDGI,GETNDP)
    IMPLICIT NONE
    INTEGER , intent(in) :: NDGI
    REAL, dimension(NDGI), intent(inout) :: WEIT, QUAPOS
    LOGICAL , intent(in) :: GETNDP
    !     This computes the weight and points for Chebyshev-Gauss-Lobatto quadrature.
    !     See page 67 of:Spectral Methods in Fluid Dynamics, C.Canuto
    !     IF(GETNDP) then get the POSITION OF THE NODES
    !     AND DONT BOTHER WITH THE WEITS.
    !Local variables
    real , PARAMETER :: PIE=3.141569254
    INTEGER :: IG,J
    !
    IF(.NOT.GETNDP) THEN
       !     THE WEIGHTS...
       WEIT(1)       =PIE/(2.*(NDGI-1))
       WEIT(NDGI)=PIE/(2.*(NDGI-1))
       do IG=2,NDGI-1
          WEIT(IG)   =PIE/(NDGI-1)
       END DO
    ENDIF
    !
    !     The quad points...
    do IG=1,NDGI
       J=IG-1
       QUAPOS(IG)=COS(PIE*REAL(J)/REAL(NDGI-1))
    END DO
  END SUBROUTINE CHEROT




  SUBROUTINE LEGROT(WEIT,QUAPOS,NDGI,GETNDP)
    IMPLICIT NONE
    !     This computes the weight and points for Chebyshev-Gauss-Lobatto quadrature.
    !     See page 69 of:Spectral Methods in Fluid Dynamics, C.Canuto
    !     IF(GETNDP) then get the POSITION OF THE NODES
    !     AND DONT BOTHER WITH THE WEITS.
    INTEGER , intent(in) :: NDGI
    REAL , dimension(NDGI) , intent(inout) :: WEIT,QUAPOS
    LOGICAL , intent(in) :: GETNDP
    !Local variables
    INTEGER :: N,IG
    real :: plegen
    !     Work out the root's i.e the quad positions first.
    CALL LROOTS(QUAPOS,NDGI)
    IF(.NOT.GETNDP) THEN
       !     THE WEIGHTS...
       N=NDGI-1
       do IG=1,NDGI
          WEIT(IG)=2./(REAL(N*(N+1))*PLEGEN(QUAPOS(IG),NDGI-1)**2)
       END DO
    ENDIF
  END SUBROUTINE LEGROT



  real function PLEGEN(LX,K)
    REAL , intent(in):: LX
    INTEGER , intent(in) :: K
    !Local variables
    REAL :: R
    INTEGER :: L
    R=0.
    DO L=0,INT(K/2)
       R=R+((-1)**L)*binomial_coefficient(K,L) &
                      *binomial_coefficient(2*K-2*L,K)*LX**(K-2*L)
    END DO
    PLEGEN=R/REAL(2**K)
  end function plegen



  real function binomial_coefficient(K,L)
    !<! Calculate binomial coefficients
    integer factorial

    INTEGER K,L
    binomial_coefficient=factorial(K)/(factorial(L)*factorial(K-L))

  end function binomial_coefficient



  SUBROUTINE LROOTS(QUAPOS,NDGI)
    IMPLICIT NONE
    INTEGER, intent(in) :: NDGI
    REAL , dimension(NDGI), intent(inout) :: QUAPOS
    !Local variables
    REAL ALPHA,BETA,RKEEP
    INTEGER N,I
    !     This sub works out the Gauss-Lobatto-Legendre roots.
    ALPHA = 0.
    BETA   = 0.
    !
    N=NDGI-1
    CALL JACOBL(N,ALPHA,BETA,QUAPOS)
    !     Now reverse ordering.
    do I=1,INT(0.1+ NDGI/2)
       RKEEP=QUAPOS(I)
       QUAPOS(I)=QUAPOS(NDGI+1-I)
       QUAPOS(NDGI+1-I)=RKEEP
    END DO
  END SUBROUTINE LROOTS




  REAL FUNCTION CHEBY1(DIFF,LX,INOD,NDNOD,NODPOS)
    IMPLICIT NONE
    INTEGER, intent(in) :: NDNOD,INOD
    REAL, intent(in) :: LX
    real, dimension(NDNOD), intent(in) :: NODPOS
    LOGICAL, intent(in)  ::  DIFF
    !Local variables
    LOGICAL  ::  DIFF2
    real :: RNX
    real tcheb
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    INTEGER NX,N

    NX=NDNOD-1
    RNX=REAL(NX)
    DIFF2=.FALSE.
    !
    CJ=1.
    IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
    HX=0.
    !
    CN=2.
    N=0
    XPT=NODPOS(INOD)
    HX=HX + ( (2./RNX)/(CJ*CN) )*TCHEB(N,XPT,.FALSE.,DIFF2)&
         &     *TCHEB(N,LX,DIFF,DIFF2)
    CN=1.
    do  N=1,NX-1! Was loop 10
       HX=HX + ( (2./RNX)/(CJ*CN) )*TCHEB(N,XPT,.FALSE.,DIFF2)&
            &        *TCHEB(N,LX,DIFF,DIFF2)
    end do ! Was loop 10
    CN=2.
    N=NX
    HX=HX + ( (2./RNX)/(CJ*CN))*TCHEB(N,XPT,.FALSE.,DIFF2)&
         &     *TCHEB(N,LX,DIFF,DIFF2)
    CHEBY1=HX
  END FUNCTION CHEBY1




  REAL FUNCTION CHEBY2(DIFF,LX,INOD,NDNOD,NODPOS)
    IMPLICIT NONE
    INTEGER , intent(in) :: NDNOD,INOD
    REAL, dimension(NDNOD), intent(in) :: NODPOS
    LOGICAL , intent(in) :: DIFF
    REAL, intent(in) :: LX
    !Local variables
    real :: R,RR,RCONST
    real :: tcheb
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    IF(.NOT.DIFF) THEN
       IF(ABS(LX-NODPOS(INOD)).LT.1.E-10) THEN
          CHEBY2=1.
       ELSE
          R=(-1.)**(INOD)*(1.-LX**2)&
               &       *TCHEB(NDNOD-1,LX,.TRUE.,.FALSE.)
          CJ=1.
          IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
          R=R/(CJ*(NDNOD-1)**2 *(LX-NODPOS(INOD)) )
          CHEBY2=R
       ENDIF
    ELSE
       IF(ABS(LX-NODPOS(INOD)).LT.1.E-10) THEN
          IF(ABS(LX+1.).LT.1.E-10) THEN
             CHEBY2= (2.*(NDNOD-1)**2+1)/6.
          ELSE
             IF(ABS(LX-1.).LT.1.E-10) THEN
                CHEBY2=-(2.*(NDNOD-1)**2+1)/6.
             ELSE
                CHEBY2=-LX/(2.*(1.-LX**2))
             ENDIF
          ENDIF
       ELSE
          R=(-1.)**(INOD)*(1.-LX**2)&
               &          *TCHEB(NDNOD-1,LX,.FALSE.,.TRUE.)
          CJ=1.
          IF((INOD.EQ.1).OR.(INOD.EQ.NDNOD)) CJ=2.
          R=R/(CJ*(NDNOD-1)**2 *(LX-NODPOS(INOD)) )
          RR=(-1.)**(INOD)*TCHEB(NDNOD-1,LX,.TRUE.,.FALSE.)&
               &           /(CJ*REAL(NDNOD-1)**2)
          RCONST=-(2*LX+(1.-LX**2)/(LX-NODPOS(INOD) ) )&
               &              /(LX-NODPOS(INOD))
          RR=RR*RCONST
          CHEBY2=RR
       ENDIF
    ENDIF
  END FUNCTION CHEBY2



  REAL FUNCTION TCHEB(N,XPT,DIFF,DIFF2)
    !      use math_utilities
    IMPLICIT NONE
    LOGICAL , intent(in) :: DIFF,DIFF2
    integer, intent(in) :: N
    real, intent(in) :: XPT
    !Local variables
    REAL :: DDT,DT,T,TTEMP,TM1,DTM1,DTEMP,R,RR
    INTEGER :: K,L
    INTEGER :: NI
    integer :: factorial
    ! If DIFF then return the n'th Chebyshef polynomial
    ! differentiated w.r.t x.
    ! If DIFF2 then form the 2'nd derivative.
    ! This sub returns the value of the K'th Chebyshef polynomial at a
    ! point XPT.
    !
    ! This formula can be found in:Spectral Methods for Fluid Dynamics, page 66
    K=N
    !
    IF((.NOT.DIFF).AND.(.NOT.DIFF2)) THEN
       IF(N.EQ.0) T=1.
       IF(N.EQ.1) T=XPT
       IF(N.GT.1) THEN
          TM1=1.
          T=XPT
          do  NI=2,N! Was loop 110
             TTEMP=T
             T=2.*XPT*TTEMP - TM1
             TM1=TTEMP
          end do ! Was loop 110
       ENDIF
       TCHEB=T
    ENDIF
    !
    IF(DIFF.AND.(.NOT.DIFF2)) THEN
       ! This part forms the differential w.r.t x.
       IF(N.EQ.0) DT=0.
       IF(N.EQ.1) THEN
          T=XPT
          DT=1.
       ENDIF
       IF(N.EQ.2) THEN
          T=2.*XPT*XPT -1
          DT=4.*XPT
       ENDIF
       IF(N.GT.2) THEN
          TM1=XPT
          DTM1=1.
          T=2.*XPT*XPT -1
          DT=4.*XPT
          do  NI=2,N! Was loop 10
             TTEMP=T
             T=2.*XPT*TTEMP - TM1
             TM1=TTEMP
             !
             DTEMP=DT
             DT=2.*XPT*DTEMP+2.*TTEMP-DTM1
             DTM1=DTEMP
          end do ! Was loop 10
       ENDIF
       TCHEB=DT
    ENDIF
    !
    IF(DIFF2) THEN
       IF(N.LE.1) THEN
          DDT=0.
       ELSE
          R=0.
          do  L=0,INT(0.1+K/2)! Was loop 50
             RR=(-1.)**K*factorial(K-L-1)/(factorial(L)*factorial(K-2*L))
             RR=RR*2**(K-2*L)
             ! The following is for 2'nd derivative.
             RR=RR*(K-2.*L)*(K-2.*L-1)*XPT**(K-2*L-2)
             R=R+RR
          end do ! Was loop 50
          DDT=R*REAL(K)*0.5
       ENDIF
       TCHEB=DDT
    ENDIF
  END FUNCTION TCHEB



  recursive function factorial(n) result(f)
    ! Calculate n!
    integer :: f
    integer, intent(in) :: n

    if (n==0) then
       f=1
    else
       f=n*factorial(n-1)
    end if

  end function factorial



  REAL FUNCTION LEGEND(DIFF,LX,INOD,NDNOD,NODPOS)
    INTEGER , intent(in) :: INOD,NDNOD
    REAL, dimension(NDNOD), intent(in) :: NODPOS
    REAL, intent(in) :: LX
    LOGICAL , intent(in) :: DIFF
    !     If DIFF then returns the spectral function DIFFERENTIATED W.R.T X
    !     associated.
    !     This function returns the spectral function associated
    !     with node INOD at POINT LX
    !     NDNOD=no of nodes in 1-D.
    !     NDGI=no of Gauss pts in 1-D.
    !     NB The nodes are at the points COS(pie*J/2.) j=0,..,ndgi-1
    REAL CJ,CN,HX,XPT
    LEGEND=1.
  END FUNCTION LEGEND



! ************************************************************************
! *********TRIANGLES AND TETS ********************************************
! ************************************************************************




  SUBROUTINE SHATRInew(L1, L2, L3, L4, WEIGHT, &
       NLOC,NGI,ndim,  N,NLX_ALL)
    ! Interface to SHATRIold using the new style variables
    IMPLICIT NONE
    INTEGER , intent(in) :: NLOC,NGI,ndim
    REAL , dimension(ngi), intent(in) :: L1, L2, L3, L4
    REAL , dimension(ngi), intent(inout) :: WEIGHT
    REAL , dimension(ngi, nloc ), intent(inout) ::N
    real, dimension (ngi,ndim,nloc), intent(inout) :: NLX_ALL
! local variables...
    REAL Nold(nloc, ngi ),NLXold(nloc, ngi ),NLYold(nloc, ngi ),NLZold(nloc, ngi )
    integer iloc

!      print *,'**going into SHATRIold...'
!      print *,'just before ndim,nloc,ngi:',ndim,nloc,ngi
!    call SHATRIold(L1, L2, L3, L4, WEIGHT, size(NLX_ALL,1)==3, &
    call SHATRIold(L1, L2, L3, L4, WEIGHT, ndim==3, &
         NLOC,NGI,  &
         Nold,NLXold,NLYold,NLZold)
!         N,NLX_ALL(:,1,:),NLX_ALL(:,2,:),NLX_ALL(:,ndim,:))
    do iloc=1,nloc
       N(:,iloc) = Nold(iloc,:)
       NLX_ALL(:,1,iloc)=NLXold(iloc,:)
       NLX_ALL(:,2,iloc)=NLYold(iloc,:)
       if(ndim==3) NLX_ALL(:,3,iloc) = NLZold(iloc,:)
    end do
!    print *,'n:',n
!    print *,'nlx_all:',nlx_all
!    print *,'nloc,ngi:',nloc,ngi
!    print *,'leaving SHATRInew'

  end subroutine SHATRInew
  !
  !
  SUBROUTINE SHATRIold(L1, L2, L3, L4, WEIGHT, D3, &
       NLOC,NGI,  &
       N,NLX,NLY,NLZ)
    ! Work out the shape functions and there derivatives...
    IMPLICIT NONE
    INTEGER , intent(in) :: NLOC,NGI
    LOGICAL , intent(in) :: D3
    REAL , dimension(ngi), intent(in) :: L1, L2, L3, L4
    REAL , dimension(ngi), intent(inout) :: WEIGHT
    REAL , dimension(nloc, ngi ), intent(inout) ::N,NLX,NLY,NLZ
    ! Local variables...
    INTEGER ::  GI
    !
    IF(.NOT.D3) THEN
       ! Assume a triangle...
       !
       IF(NLOC.EQ.1) THEN
          Loop_Gi_Nloc1: DO GI=1,NGI
             N(1,GI)=1.0
             NLX(1,GI)=0.0
             NLY(1,GI)=0.0
          end DO Loop_Gi_Nloc1
       ELSE IF((NLOC.EQ.3).OR.(NLOC.EQ.4)) THEN
          Loop_Gi_Nloc3_4: DO GI=1,NGI
             N(1,GI)=L1(GI)
             N(2,GI)=L2(GI)
             N(3,GI)=L3(GI)
             !
             NLX(1,GI)=1.0
             NLX(2,GI)=0.0
             NLX(3,GI)=-1.0
             !
             NLY(1,GI)=0.0
             NLY(2,GI)=1.0
             NLY(3,GI)=-1.0
             IF(NLOC.EQ.4) THEN
                ! Bubble function...
                !alpha == 1 behaves better than the correct value of 27. See Osman et al. 2019
                N(4,GI)  =1. * L1(GI)*L2(GI)*L3(GI)
                NLX(4,GI)=1. * L2(GI)*(1.-L2(GI))-2.*L1(GI)*L2(GI)
                NLY(4,GI)=1. * L1(GI)*(1.-L1(GI))-2.*L1(GI)*L2(GI)
             ENDIF
          end DO Loop_Gi_Nloc3_4
       ELSE IF((NLOC.EQ.6).OR.(NLOC.EQ.7)) THEN
          Loop_Gi_Nloc_6_7: DO GI=1,NGI
             N(1,GI)=(2.*L1(GI)-1.)*L1(GI)
             N(2,GI)=(2.*L2(GI)-1.)*L2(GI)
             N(3,GI)=(2.*L3(GI)-1.)*L3(GI)
             !
             N(4,GI)=4.*L1(GI)*L2(GI)
             N(5,GI)=4.*L2(GI)*L3(GI)
             N(6,GI)=4.*L1(GI)*L3(GI)

             !
             ! nb L1+L2+L3+L4=1
             ! x-derivative...
             NLX(1,GI)=4.*L1(GI)-1.
             NLX(2,GI)=0.
             NLX(3,GI)=-4.*(1.-L2(GI))+4.*L1(GI) + 1.
             !
             NLX(4,GI)=4.*L2(GI)
             NLX(5,GI)=-4.*L2(GI)
             NLX(6,GI)=4.*(1.-L2(GI))-8.*L1(GI)
             !
             ! y-derivative...
             NLY(1,GI)=0.
             NLY(2,GI)=4.*L2(GI)-1.0
             NLY(3,GI)=-4.*(1.-L1(GI))+4.*L2(GI) + 1.
             !
             NLY(4,GI)=4.*L1(GI)
             NLY(5,GI)=4.*(1.-L1(GI))-8.*L2(GI)
             NLY(6,GI)=-4.*L1(GI)
             IF(NLOC.EQ.7) THEN
                ! Bubble function...
                N(7,GI)  =L1(GI)*L2(GI)*L3(GI)
                NLX(7,GI)=L2(GI)*(1.-L2(GI))-2.*L1(GI)*L2(GI)
                NLY(7,GI)=L1(GI)*(1.-L1(GI))-2.*L1(GI)*L2(GI)
             ENDIF
          END DO Loop_Gi_Nloc_6_7
          ! ENDOF IF(NLOC.EQ.6) THEN...
       ELSE IF(NLOC==10) THEN ! Cubic triangle...
          ! get the shape functions for a cubic triangle...
          call shape_triangle_cubic( l1, l2, l3, l4, weight, d3, &
               nloc, ngi, &
               n, nlx, nly, nlz )

       ELSE ! has not found the element shape functions
          stop 811
       ENDIF
       !
       ! ENDOF IF(.NOT.D3) THEN
    ENDIF
    !
    !
    IF(D3) THEN
       ! Assume a tet...
       ! This is for 5 point quadrature.
       IF((NLOC.EQ.10).OR.(NLOC.EQ.11)) THEN
          Loop_Gi_Nloc_10_11: DO GI=1,NGI
             !ewrite(3,*)'gi,L1(GI),L2(GI),L3(GI),L4(GI):',gi,L1(GI),L2(GI),L3(GI),L4(GI)
             N(1,GI)=(2.*L1(GI)-1.)*L1(GI)
             N(3,GI)=(2.*L2(GI)-1.)*L2(GI)
             N(5,GI)=(2.*L3(GI)-1.)*L3(GI)
             N(10,GI)=(2.*L4(GI)-1.)*L4(GI)

             !if(L1(GI).gt.-1.93) ewrite(3,*)'gi,L1(GI), L2(GI), L3(GI), L4(GI),N(1,GI):', &
             !                            gi,L1(GI), L2(GI), L3(GI), L4(GI),N(1,GI)
             !
             !
             N(2,GI)=4.*L1(GI)*L2(GI)
             N(6,GI)=4.*L1(GI)*L3(GI)
             N(7,GI)=4.*L1(GI)*L4(GI)
             !
             N(4,GI) =4.*L2(GI)*L3(GI)
             N(9,GI) =4.*L3(GI)*L4(GI)
             N(8,GI)=4.*L2(GI)*L4(GI)
             ! nb L1+L2+L3+L4=1
             ! x-derivative...
             NLX(1,GI)=4.*L1(GI)-1.
             NLX(3,GI)=0.
             NLX(5,GI)=0.
             NLX(10,GI)=-4.*(1.-L2(GI)-L3(GI))+4.*L1(GI) + 1.
             !if(L1(GI).gt.-1.93) ewrite(3,*)'Nlx(1,GI):', &
             !     Nlx(1,GI)
             !
             NLX(2,GI)=4.*L2(GI)
             NLX(6,GI)=4.*L3(GI)
             NLX(7,GI)=4.*(L4(GI)-L1(GI))
             !
             NLX(4,GI) =0.
             NLX(9,GI) =-4.*L3(GI)
             NLX(8,GI)=-4.*L2(GI)
             !
             ! y-derivative...
             NLY(1,GI)=0.
             NLY(3,GI)=4.*L2(GI)-1.0
             NLY(5,GI)=0.
             NLY(10,GI)=-4.*(1.-L1(GI)-L3(GI))+4.*L2(GI) + 1.
             !
             NLY(2,GI)=4.*L1(GI)
             NLY(6,GI)=0.
             NLY(7,GI)=-4.*L1(GI)
             !
             NLY(4,GI) =4.*L3(GI)
             NLY(9,GI) =-4.*L3(GI)
             NLY(8,GI)=4.*(1-L1(GI)-L3(GI))-8.*L2(GI)
             !
             ! z-derivative...
             NLZ(1,GI)=0.
             NLZ(3,GI)=0.
             NLZ(5,GI)=4.*L3(GI)-1.
             NLZ(10,GI)=-4.*(1.-L1(GI)-L2(GI))+4.*L3(GI) + 1.
             !
             NLZ(2,GI)=0.
             NLZ(6,GI)=4.*L1(GI)
             NLZ(7,GI)=-4.*L1(GI)
             !
             NLZ(4,GI) =4.*L2(GI)
             NLZ(9,GI) =4.*(1.-L1(GI)-L2(GI))-8.*L3(GI)
             NLZ(8,GI)=-4.*L2(GI)
             IF(NLOC.EQ.11) THEN
                ! Bubble function...
                N(11,GI)  =L1(GI)*L2(GI)*L3(GI)*L4(GI)
                NLX(11,GI)=L2(GI)*L3(GI)*(1.-L2(GI)-L3(GI))-2.*L1(GI)*L2(GI)*L3(GI)
                NLY(11,GI)=L1(GI)*L3(GI)*(1.-L1(GI)-L3(GI))-2.*L1(GI)*L2(GI)*L3(GI)
                NLZ(11,GI)=L1(GI)*L2(GI)*(1.-L1(GI)-L2(GI))-2.*L1(GI)*L2(GI)*L3(GI)
             ENDIF
             !
          end DO Loop_Gi_Nloc_10_11
          ! ENDOF IF(NLOC.EQ.10) THEN...
       ENDIF
       !
       IF((NLOC.EQ.4).OR.(NLOC.EQ.5)) THEN
          Loop_Gi_Nloc_4_5: DO GI=1,NGI
             N(1,GI)=L1(GI)
             N(2,GI)=L2(GI)
             N(3,GI)=L3(GI)
             N(4,GI)=L4(GI)
             !
             NLX(1,GI)=1.0
             NLX(2,GI)=0
             NLX(3,GI)=0
             NLX(4,GI)=-1.0
             !
             NLY(1,GI)=0.0
             NLY(2,GI)=1.0
             NLY(3,GI)=0.0
             NLY(4,GI)=-1.0
             !
             NLZ(1,GI)=0.0
             NLZ(2,GI)=0.0
             NLZ(3,GI)=1.0
             NLZ(4,GI)=-1.0
             IF(NLOC.EQ.5) THEN
                ! Bubble function ...
                !alpha == 50 behaves better than the correct value of 256. See Osman et al. 2019
                N(5,GI)  = 50. * L1(GI)*L2(GI)*L3(GI)*L4(GI)
                NLX(5,GI)= 50. * L2(GI)*L3(GI)*(1.-L2(GI)-L3(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
                NLY(5,GI)= 50. * L1(GI)*L3(GI)*(1.-L1(GI)-L3(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
                NLZ(5,GI)= 50. * L1(GI)*L2(GI)*(1.-L1(GI)-L2(GI))  &
                         -2.*L1(GI)*L2(GI)*L3(GI)
             ENDIF
          end DO Loop_Gi_Nloc_4_5
       ENDIF
       !
       IF(NLOC.EQ.1) THEN
          Loop_Gi_Nloc_1: DO GI=1,NGI
             N(1,GI)=1.0
             NLX(1,GI)=0.0
             NLY(1,GI)=0.0
             NLZ(1,GI)=0.0
          end DO Loop_Gi_Nloc_1
       ENDIF
       !
       ! ENDOF IF(D3) THEN...
    ENDIF
    !
    RETURN
  END SUBROUTINE SHATRIold
  !
  !
  !
  !
  SUBROUTINE TRIQUAold(L1, L2, L3, L4, WEIGHT, D3,NGI)
    ! This sub calculates the local corrds L1, L2, L3, L4 and
    ! weights at the quadrature points.
    ! If D3 it does this for 3Dtetrahedra elements else
    ! triangular elements.
    IMPLICIT NONE
    INTEGER , intent(in):: NGI
    LOGICAL , intent(in) :: D3
    REAL , dimension(ngi) , intent(inout) ::L1, L2, L3, L4, WEIGHT
    ! Local variables...
    REAL :: ALPHA,BETA
    REAL :: ALPHA1,BETA1
    REAL :: ALPHA2,BETA2
    real :: rsum
    INTEGER I
    !
    IF(D3) THEN
       ! this is for a tetrahedra element...
       ! This is for one point.
       IF(NGI.EQ.1) THEN
          ! Degree of precision is 1
          DO I=1,NGI
             L1(I)=0.25
             L2(I)=0.25
             L3(I)=0.25
             L4(I)=0.25
             WEIGHT(I)=1.0
          END DO
       ENDIF

       IF(NGI.EQ.4) THEN
          ! Degree of precision is 2
          ALPHA=0.58541020
          BETA=0.13819660
          DO I=1,NGI
             L1(I)=BETA
             L2(I)=BETA
             L3(I)=BETA
             L4(I)=BETA
             WEIGHT(I)=0.25
          END DO
          L1(1)=ALPHA
          L2(2)=ALPHA
          L3(3)=ALPHA
          L4(4)=ALPHA
       ENDIF

       IF(NGI.EQ.5) THEN
          ! Degree of precision is 3
          L1(1)=0.25
          L2(1)=0.25
          L3(1)=0.25
          L4(1)=0.25
          WEIGHT(1)=-4./5.
          !
          DO I=2,NGI
             L1(I)=1./6.
             L2(I)=1./6.
             L3(I)=1./6.
             L4(I)=1./6.
             WEIGHT(I)=9./20.
          END DO
          L1(2)=0.5
          L2(3)=0.5
          L3(4)=0.5
          L4(5)=0.5
       ENDIF
       !
       IF(NGI.EQ.11) THEN
          ! Degree of precision is 4
          ALPHA=(1.+SQRT(5./14.))/4.0
          BETA =(1.-SQRT(5./14.))/4.0
          I=1
          L1(I)=0.25
          L2(I)=0.25
          L3(I)=0.25
          WEIGHT(I)=-6.*74.0/5625.0
          DO I=2,5
             L1(I)=1./14.
             L2(I)=1./14.
             L3(I)=1./14.
             WEIGHT(I)=6.*343./45000.
          END DO
          L1(2)=11./14.
          L2(3)=11./14.
          L3(4)=11./14.
          DO I=6,11
             L1(I)=ALPHA
             L2(I)=ALPHA
             L3(I)=ALPHA
             WEIGHT(I)=6.*56.0/2250.0
          END DO
          L3(6)=BETA
          L2(7)=BETA
          L2(8)=BETA
          L3(8)=BETA
          L1(9)=BETA
          L1(10)=BETA
          L3(10)=BETA
          L1(11)=BETA
          L2(11)=BETA
          ! ENDOF IF(NGI.EQ.11) THEN...
       ENDIF

        if (NGI == 15) then!Fith order quadrature
          ! Degree of precision is 5
         L1=(/0.2500000000000000, 0.0000000000000000, 0.3333333333333333, &
             0.3333333333333333, 0.3333333333333333, &
             0.7272727272727273, 0.0909090909090909, 0.0909090909090909, &
             0.0909090909090909, 0.4334498464263357, &
             0.0665501535736643, 0.0665501535736643, 0.0665501535736643, &
             0.4334498464263357, 0.4334498464263357/)
         L2=(/0.2500000000000000, 0.3333333333333333, 0.3333333333333333, &
             0.3333333333333333, 0.0000000000000000, &
             0.0909090909090909, 0.0909090909090909, 0.0909090909090909, &
             0.7272727272727273, 0.0665501535736643, &
             0.4334498464263357, 0.0665501535736643, 0.4334498464263357, &
             0.0665501535736643, 0.4334498464263357/)
         L3=(/0.2500000000000000, 0.3333333333333333, 0.3333333333333333, &
             0.0000000000000000, 0.3333333333333333, &
             0.0909090909090909, 0.0909090909090909, 0.7272727272727273, &
             0.0909090909090909, 0.0665501535736643, &
             0.0665501535736643, 0.4334498464263357, 0.4334498464263357, &
             0.4334498464263357, 0.0665501535736643/)
         !We divide the weights later by 6
         WEIGHT=(/0.1817020685825351, 0.0361607142857143, 0.0361607142857143, &
             0.0361607142857143, 0.0361607142857143, &
             0.0698714945161738, 0.0698714945161738, 0.0698714945161738, &
             0.0698714945161738, 0.0656948493683187, &
             0.0656948493683187, 0.0656948493683187, 0.0656948493683187, &
             0.0656948493683187, 0.0656948493683187/)
       end if


       if (NGI == 45) then!Eighth order quadrature, for bubble shape functions or P3
         !Obtained from: https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
         !Here to get triangle quadrature sets:
         !https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
         ! Degree of precision is 8
        L1=(/0.2500000000000000,0.6175871903000830,0.1274709365666390,0.1274709365666390,&
        0.1274709365666390,0.9037635088221031,&
        0.0320788303926323,0.0320788303926323,0.0320788303926323,0.4502229043567190,&
        0.0497770956432810,0.0497770956432810,&
        0.0497770956432810,0.4502229043567190,0.4502229043567190,0.3162695526014501,&
        0.1837304473985499,0.1837304473985499,&
        0.1837304473985499,0.3162695526014501,0.3162695526014501,0.0229177878448171,&
        0.2319010893971509,0.2319010893971509,&
        0.5132800333608811,0.2319010893971509,0.2319010893971509,0.2319010893971509,&
        0.0229177878448171,0.5132800333608811,&
        0.2319010893971509,0.0229177878448171,0.5132800333608811,0.7303134278075384,&
        0.0379700484718286,0.0379700484718286,&
        0.1937464752488044,0.0379700484718286,0.0379700484718286,0.0379700484718286,&
        0.7303134278075384,0.1937464752488044,&
        0.0379700484718286,0.7303134278075384,0.1937464752488044/)
        L2=(/0.2500000000000000,0.1274709365666390,0.1274709365666390,0.1274709365666390,&
        0.6175871903000830,0.0320788303926323,&
        0.0320788303926323,0.0320788303926323,0.9037635088221031,0.0497770956432810,&
        0.4502229043567190,0.0497770956432810,&
        0.4502229043567190,0.0497770956432810,0.4502229043567190,0.1837304473985499,&
        0.3162695526014501,0.1837304473985499,&
        0.3162695526014501,0.1837304473985499,0.3162695526014501,0.2319010893971509,&
        0.0229177878448171,0.2319010893971509,&
        0.2319010893971509,0.5132800333608811,0.2319010893971509,0.0229177878448171,&
        0.5132800333608811,0.2319010893971509,&
        0.5132800333608811,0.2319010893971509,0.0229177878448171,0.0379700484718286,&
        0.7303134278075384,0.0379700484718286,&
        0.0379700484718286,0.1937464752488044,0.0379700484718286,0.7303134278075384,&
        0.1937464752488044,0.0379700484718286,&
        0.1937464752488044,0.0379700484718286,0.7303134278075384/)
        L3=(/0.2500000000000000,0.1274709365666390,0.1274709365666390,0.6175871903000830,&
        0.1274709365666390,0.0320788303926323,&
        0.0320788303926323,0.9037635088221031,0.0320788303926323,0.0497770956432810,&
        0.0497770956432810,0.4502229043567190,&
        0.4502229043567190,0.4502229043567190,0.0497770956432810,0.1837304473985499,&
        0.1837304473985499,0.3162695526014501,&
        0.3162695526014501,0.3162695526014501,0.1837304473985499,0.2319010893971509,&
        0.2319010893971509,0.0229177878448171,&
        0.2319010893971509,0.2319010893971509,0.5132800333608811,0.5132800333608811,&
        0.2319010893971509,0.0229177878448171,&
        0.0229177878448171,0.5132800333608811,0.2319010893971509,0.0379700484718286,&
        0.0379700484718286,0.7303134278075384,&
        0.0379700484718286,0.0379700484718286,0.1937464752488044,0.1937464752488044,&
        0.0379700484718286,0.7303134278075384,&
        0.7303134278075384,0.1937464752488044,0.0379700484718286/)
        !We divide the weights later by 6
        WEIGHT=(/-0.2359620398477557,0.0244878963560562,0.0244878963560562,0.0244878963560562,&
        0.0244878963560562,0.0039485206398261,&
        0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0263055529507371,&
        0.0263055529507371,0.0263055529507371,&
        0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0829803830550589,&
        0.0829803830550589,0.0829803830550589,&
        0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,&
        0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,&
        0.0134324384376852,0.0134324384376852,0.0134324384376852/)
      end if

       DO I=1,NGI
          L4(I)=1.0-L1(I)-L2(I)-L3(I)
       END DO

       ! Now multiply by 1/6. to get weigts correct...
       DO I=1,NGI
          WEIGHT(I)=WEIGHT(I)/6.
       END DO
       ! ENDOF IF(D3) THEN...
    ENDIF
    !
    IF(.NOT.D3) THEN
       ! 2-D TRAINGULAR ELEMENTS...
       IF(NGI.EQ.1) THEN
          ! LINEAR
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=1.0
!          WEIGHT(I)=0.5 !************************ISSUE WITH WEIGHTS
       ENDIF
       !
       IF(NGI.EQ.3) THEN
          ! QUADRASTIC
          DO I=1,NGI
             L1(I)=0.5
             L2(I)=0.5
             WEIGHT(I)=1.0/3.0
          END DO
          L1(2)=0.0
          L2(3)=0.0
       ENDIF
       !
       IF(NGI.EQ.4) THEN
          ! CUBIC
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=-27./48.
          DO I=2,NGI
             L1(I)=0.2
             L2(I)=0.2
             WEIGHT(I)=25./48.
          END DO
          L1(1)=0.6
          L2(2)=0.6
       ENDIF
       !
       IF(NGI.EQ.7) THEN
          ! QUNTIC
          ALPHA1=0.0597158717
          BETA1 =0.4701420641
          ALPHA2=0.7974269853
          BETA2 =0.1012865073
          I=1
          L1(I)=1./3.
          L2(I)=1./3.
          WEIGHT(I)=0.225
          DO I=2,4
             L1(I)=BETA1
             L2(I)=BETA1
             WEIGHT(I)=0.1323941527
          END DO
          L1(2)=ALPHA1
          L2(4)=ALPHA1
          DO I=5,7
             L1(I)=BETA2
             L2(I)=BETA2
             WEIGHT(I)=0.1259391805
          END DO
          L1(5)=ALPHA2
          L2(6)=ALPHA2
          ! ENDOF IF(NGI.EQ.7) THEN...
       ENDIF

       IF(NGI.EQ.14) THEN
          ! 5th order quadrature set...
          L1(1) = 6.943184420297371E-002
          L1(2) = 6.943184420297371E-002
          L1(3) = 6.943184420297371E-002
          L1(4) = 6.943184420297371E-002
          L1(5) = 6.943184420297371E-002
          L1(6) = 0.330009478207572
          L1(7) = 0.330009478207572
          L1(8) = 0.330009478207572
          L1(9) = 0.330009478207572
          L1(10) = 0.669990521792428
          L1(11) = 0.669990521792428
          L1(12) = 0.669990521792428
          L1(13) = 0.930568155797026
          L1(14) = 0.930568155797026
          ! local coord 1:
          L2(1) = 4.365302387072518E-002
          L2(2) = 0.214742881469342
          L2(3) = 0.465284077898513
          L2(4) = 0.715825274327684
          L2(5) = 0.886915131926301
          L2(6) = 4.651867752656094E-002
          L2(7) = 0.221103222500738
          L2(8) = 0.448887299291690
          L2(9) = 0.623471844265867
          L2(10) = 3.719261778493340E-002
          L2(11) = 0.165004739103786
          L2(12) = 0.292816860422638
          L2(13) = 1.467267513102734E-002
          L2(14) = 5.475916907194637E-002
          ! local coord 2:
          WEIGHT(1) = 1.917346464706755E-002
          WEIGHT(2) = 3.873334126144628E-002
          WEIGHT(3) = 4.603770904527855E-002
          WEIGHT(4) = 3.873334126144628E-002
          WEIGHT(5) = 1.917346464706755E-002
          WEIGHT(6) = 3.799714764789616E-002
          WEIGHT(7) = 7.123562049953998E-002
          WEIGHT(8) = 7.123562049953998E-002
          WEIGHT(9) = 3.799714764789616E-002
          WEIGHT(10) = 2.989084475992800E-002
          WEIGHT(11) = 4.782535161588505E-002
          WEIGHT(12) = 2.989084475992800E-002
          WEIGHT(13) = 6.038050853208200E-003
          WEIGHT(14) = 6.038050853208200E-003
          rsum=SUM(WEIGHT(1:NGI))
          WEIGHT(1:NGI)=WEIGHT(1:NGI)/RSUM
          ! ENDOF IF(NGI.EQ.14) THEN...
       ENDIF
! 
       if(ngi==44) then ! (n=9)
          L1(1) = 1.985507175123191E-002
          L1(2) = 1.985507175123191E-002
          L1(3) = 1.985507175123191E-002
          L1(4) = 1.985507175123191E-002
          L1(5) = 1.985507175123191E-002
          L1(6) = 1.985507175123191E-002
          L1(7) = 1.985507175123191E-002
          L1(8) = 1.985507175123191E-002
          L1(9) = 1.985507175123191E-002
          L1(10) = 0.101666761293187
          L1(11) = 0.101666761293187
          L1(12) = 0.101666761293187
          L1(13) = 0.101666761293187
          L1(14) = 0.101666761293187
          L1(15) = 0.101666761293187
          L1(16) = 0.101666761293187
          L1(17) = 0.101666761293187
          L1(18) = 0.237233795041836
          L1(19) = 0.237233795041836
          L1(20) = 0.237233795041836
          L1(21) = 0.237233795041836
          L1(22) = 0.237233795041836
          L1(23) = 0.237233795041836
          L1(24) = 0.237233795041836
          L1(25) = 0.408282678752175
          L1(26) = 0.408282678752175
          L1(27) = 0.408282678752175
          L1(28) = 0.408282678752175
          L1(29) = 0.408282678752175
          L1(30) = 0.408282678752175
          L1(31) = 0.591717321247825
          L1(32) = 0.591717321247825
          L1(33) = 0.591717321247825
          L1(34) = 0.591717321247825
          L1(35) = 0.591717321247825
          L1(36) = 0.762766204958164
          L1(37) = 0.762766204958164
          L1(38) = 0.762766204958164
          L1(39) = 0.762766204958164
          L1(40) = 0.898333238706813
          L1(41) = 0.898333238706813
          L1(42) = 0.898333238706813
          L1(43) = 0.980144928248768
          L1(44) = 0.980144928248768

!L2(1)...
          L2(1) = 1.560378988162790E-002
          L2(2) = 8.035663927218221E-002
          L2(3) = 0.189476014677302
          L2(4) = 0.331164789916112
          L2(5) = 0.490072464124384
          L2(6) = 0.648980138332656
          L2(7) = 0.790668913571466
          L2(8) = 0.899788288976586
          L2(9) = 0.964541138367140
          L2(10) = 1.783647091104033E-002
          L2(11) = 9.133063094134081E-002
          L2(12) = 0.213115003430640
          L2(13) = 0.366773901111335
          L2(14) = 0.531559337595478
          L2(15) = 0.685218235276173
          L2(16) = 0.807002607765473
          L2(17) = 0.880496767795773
          L2(18) = 1.940938228235618E-002
          L2(19) = 9.857563833019303E-002
          L2(20) = 0.226600619520678
          L2(21) = 0.381383102479082
          L2(22) = 0.536165585437487
          L2(23) = 0.664190566627971
          L2(24) = 0.743356822675808
          L2(25) = 1.997947907913758E-002
          L2(26) = 0.100234137152044
          L2(27) = 0.225261107830170
          L2(28) = 0.366456213417655
          L2(29) = 0.491483184095780
          L2(30) = 0.571737842168687
          L2(31) = 1.915257191055202E-002
          L2(32) = 9.421749319819557E-002
          L2(33) = 0.204141339376088
          L2(34) = 0.314065185553979
          L2(35) = 0.389130106841623
          L2(36) = 1.647157989702492E-002
          L2(37) = 7.828940091495819E-002
          L2(38) = 0.158944394126877
          L2(39) = 0.220762215144811
          L2(40) = 1.145801331145764E-002
          L2(41) = 5.083338064659329E-002
          L2(42) = 9.020874798172894E-002
          L2(43) = 4.195870365439417E-003
          L2(44) = 1.565920138579250E-002

! WEIGHT(1)...
          WEIGHT(1) = 2.015983497663207E-003
          WEIGHT(2) = 4.480916044841641E-003
          WEIGHT(3) = 6.464359484621604E-003
          WEIGHT(4) = 7.747662769908149E-003
          WEIGHT(5) = 8.191474625434276E-003
          WEIGHT(6) = 7.747662769908149E-003
          WEIGHT(7) = 6.464359484621604E-003
          WEIGHT(8) = 4.480916044841641E-003
          WEIGHT(9) = 2.015983497663207E-003
          WEIGHT(10) = 5.055663745070170E-003
          WEIGHT(11) = 1.110639128725685E-002
          WEIGHT(12) = 1.566747257514398E-002
          WEIGHT(13) = 1.811354111938598E-002
          WEIGHT(14) = 1.811354111938598E-002
          WEIGHT(15) = 1.566747257514398E-002
          WEIGHT(16) = 1.110639128725685E-002
          WEIGHT(17) = 5.055663745070170E-003
          WEIGHT(18) = 7.745946956361961E-003
          WEIGHT(19) = 1.673231410555364E-002
          WEIGHT(20) = 2.284153446586376E-002
          WEIGHT(21) = 2.500282281756943E-002
          WEIGHT(22) = 2.284153446586376E-002
          WEIGHT(23) = 1.673231410555364E-002
          WEIGHT(24) = 7.745946956361961E-003
          WEIGHT(25) = 9.191827856850984E-003
          WEIGHT(26) = 1.935542449754594E-002
          WEIGHT(27) = 2.510431683577024E-002
          WEIGHT(28) = 2.510431683577024E-002
          WEIGHT(29) = 1.935542449754594E-002
          WEIGHT(30) = 9.191827856850984E-003
          WEIGHT(31) = 8.770885597453929E-003
          WEIGHT(32) = 1.771853503082167E-002
          WEIGHT(33) = 2.105991205229386E-002
          WEIGHT(34) = 1.771853503082167E-002
          WEIGHT(35) = 8.770885597453929E-003
          WEIGHT(36) = 6.471997505236908E-003
          WEIGHT(37) = 1.213345702759751E-002
          WEIGHT(38) = 1.213345702759751E-002
          WEIGHT(39) = 6.471997505236908E-003
          WEIGHT(40) = 3.140105492486528E-003
          WEIGHT(41) = 5.024168787978471E-003
          WEIGHT(42) = 3.140105492486528E-003
          WEIGHT(43) = 5.024749628293684E-004
          WEIGHT(44) = 5.024749628293684E-004
       endif ! if(ngi=44) then
! 
       if(ngi==49) then ! (n=7)
          L1(1) = 0.2500000000D+00
          L1(2) = 0.1485387122D+00
          L1(3) = 0.3514612878D+00
          L1(4) = 0.6461720360D-01
          L1(5) = 0.4353827964D+00
          L1(6) = 0.1272302191D-01
          L1(7) = 0.4872769781D+00
          L1(8) = 0.3514612878D+00
          L1(9) = 0.2088224283D+00
          L1(10) = 0.4941001474D+00
          L1(11) = 0.9084178238D-01
          L1(12) = 0.6120807933D+00
          L1(13) = 0.1788659867D-01
          L1(14) = 0.6850359770D+00
          L1(15) = 0.1485387122D+00
          L1(16) = 0.8825499604D-01
          L1(17) = 0.2088224283D+00
          L1(18) = 0.3839262482D-01
          L1(19) = 0.2586847995D+00
          L1(20) = 0.7559445160D-02
          L1(21) = 0.2895179792D+00
          L1(22) = 0.4353827964D+00
          L1(23) = 0.2586847995D+00
          L1(24) = 0.6120807933D+00
          L1(25) = 0.1125328752D+00
          L1(26) = 0.7582327176D+00
          L1(27) = 0.2215753944D-01
          L1(28) = 0.8486080534D+00
          L1(29) = 0.6461720360D-01
          L1(30) = 0.3839262482D-01
          L1(31) = 0.9084178238D-01
          L1(32) = 0.1670153200D-01
          L1(33) = 0.1125328752D+00
          L1(34) = 0.3288504390D-02
          L1(35) = 0.1259459028D+00
          L1(36) = 0.4872769781D+00
          L1(37) = 0.2895179792D+00
          L1(38) = 0.6850359770D+00
          L1(39) = 0.1259459028D+00
          L1(40) = 0.8486080534D+00
          L1(41) = 0.2479854268D-01
          L1(42) = 0.9497554135D+00
          L1(43) = 0.1272302191D-01
          L1(44) = 0.7559445160D-02
          L1(45) = 0.1788659867D-01
          L1(46) = 0.3288504390D-02
          L1(47) = 0.2215753944D-01
          L1(48) = 0.6475011465D-03
          L1(49) = 0.2479854268D-01
! 
! L2(1) = 
          L2(1) = 0.5000000000D+00
          L2(2) = 0.7029225757D+00
          L2(3) = 0.2970774243D+00
          L2(4) = 0.8707655928D+00
          L2(5) = 0.1292344072D+00
          L2(6) = 0.9745539562D+00
          L2(7) = 0.2544604383D-01
          L2(8) = 0.5000000000D+00
          L2(9) = 0.7029225757D+00
          L2(10) = 0.2970774243D+00
          L2(11) = 0.8707655928D+00
          L2(12) = 0.1292344072D+00
          L2(13) = 0.9745539562D+00
          L2(14) = 0.2544604383D-01
          L2(15) = 0.5000000000D+00
          L2(16) = 0.7029225757D+00
          L2(17) = 0.2970774243D+00
          L2(18) = 0.8707655928D+00
          L2(19) = 0.1292344072D+00
          L2(20) = 0.9745539562D+00
          L2(21) = 0.2544604383D-01
          L2(22) = 0.5000000000D+00
          L2(23) = 0.7029225757D+00
          L2(24) = 0.2970774243D+00
          L2(25) = 0.8707655928D+00
          L2(26) = 0.1292344072D+00
          L2(27) = 0.9745539562D+00
          L2(28) = 0.2544604383D-01
          L2(29) = 0.5000000000D+00
          L2(30) = 0.7029225757D+00
          L2(31) = 0.2970774243D+00
          L2(32) = 0.8707655928D+00
          L2(33) = 0.1292344072D+00
          L2(34) = 0.9745539562D+00
          L2(35) = 0.2544604383D-01
          L2(36) = 0.5000000000D+00
          L2(37) = 0.7029225757D+00
          L2(38) = 0.2970774243D+00
          L2(39) = 0.8707655928D+00
          L2(40) = 0.1292344072D+00
          L2(41) = 0.9745539562D+00
          L2(42) = 0.2544604383D-01
          L2(43) = 0.5000000000D+00
          L2(44) = 0.7029225757D+00
          L2(45) = 0.2970774243D+00
          L2(46) = 0.8707655928D+00
          L2(47) = 0.1292344072D+00
          L2(48) = 0.9745539562D+00
          L2(49) = 0.2544604383D-01
! 
! WEIGHT(1)...
          WEIGHT(1) = 0.2183621219D-01
          WEIGHT(2) = 0.1185259869D-01
          WEIGHT(3) = 0.2804474024D-01
          WEIGHT(4) = 0.3777048400D-02
          WEIGHT(5) = 0.2544928909D-01
          WEIGHT(6) = 0.3442812316D-03
          WEIGHT(7) = 0.1318557174D-01
          WEIGHT(8) = 0.1994866947D-01
          WEIGHT(9) = 0.1082804890D-01
          WEIGHT(10) = 0.2562052651D-01
          WEIGHT(11) = 0.3450556783D-02
          WEIGHT(12) = 0.2324942860D-01
          WEIGHT(13) = 0.3145212381D-03
          WEIGHT(14) = 0.1204579851D-01
          WEIGHT(15) = 0.1994866947D-01
          WEIGHT(16) = 0.1082804890D-01
          WEIGHT(17) = 0.2562052651D-01
          WEIGHT(18) = 0.3450556783D-02
          WEIGHT(19) = 0.2324942860D-01
          WEIGHT(20) = 0.3145212381D-03
          WEIGHT(21) = 0.1204579851D-01
          WEIGHT(22) = 0.1461316874D-01
          WEIGHT(23) = 0.7931962886D-02
          WEIGHT(24) = 0.1876802249D-01
          WEIGHT(25) = 0.2527665748D-02
          WEIGHT(26) = 0.1703110194D-01
          WEIGHT(27) = 0.2303989213D-03
          WEIGHT(28) = 0.8824011376D-02
          WEIGHT(29) = 0.1461316874D-01
          WEIGHT(30) = 0.7931962886D-02
          WEIGHT(31) = 0.1876802249D-01
          WEIGHT(32) = 0.2527665748D-02
          WEIGHT(33) = 0.1703110194D-01
          WEIGHT(34) = 0.2303989213D-03
          WEIGHT(35) = 0.8824011376D-02
          WEIGHT(36) = 0.6764926484D-02
          WEIGHT(37) = 0.3671971955D-02
          WEIGHT(38) = 0.8688347794D-02
          WEIGHT(39) = 0.1170141347D-02
          WEIGHT(40) = 0.7884268950D-02
          WEIGHT(41) = 0.1066593969D-03
          WEIGHT(42) = 0.4084931154D-02
          WEIGHT(43) = 0.6764926484D-02
          WEIGHT(44) = 0.3671971955D-02
          WEIGHT(45) = 0.8688347794D-02
          WEIGHT(46) = 0.1170141347D-02
          WEIGHT(47) = 0.7884268950D-02
          WEIGHT(48) = 0.1066593969D-03
          WEIGHT(49) = 0.4084931154D-02
       endif ! if(ngi=49) then
       !
       DO I=1,NGI
          L3(I)=1.0-L1(I)-L2(I)
       END DO
       ! ENDOF IF(.NOT.D3) THEN...
    ENDIF
    weight=0.5*weight ! *********big change
    !
    RETURN
  END subroutine TRIQUAold
  !
  !



  subroutine shape_triangle_cubic( l1, l2, l3, l4, weight, d3, &
       nloc, ngi, &
       n, nlx, nly, nlz )
    implicit none
    integer, intent( in ) :: nloc, ngi
    real, dimension( ngi ), intent( in ) :: l1, l2, l3, l4, weight
    logical, intent( in ) :: d3
    real, dimension( nloc, ngi ), intent( inout ) :: n, nlx, nly, nlz
    ! Local variables
    logical :: base_order
    integer :: gi, ndim, cv_ele_type_dummy, u_nloc_dummy
    real, dimension( :, : ), allocatable :: cvn_dummy, un_dummy, unlx_dummy, &
         unly_dummy, unlz_dummy
    real, dimension( : ), allocatable :: cvweigh_dummy
    real :: a,b

    !ewrite(3,*)'In shatri d3,nloc=',d3,nloc
    if(nloc.ne.10) then ! wrong element type
       stop 28213
    endif
! The node the numbering is:
            !  3
            !  | \
            !  8  7
            !  |   \
            !  9 10 6
            !  |     \
            !  1-4-5--2

    ! cubic triangle...
    do gi = 1, ngi
       ! corner nodes...
       n( 1, gi ) = 0.5*( 3. * l1( gi ) - 1. ) * (3. * l1( gi )   -2.) *  l1( gi )
       n( 2, gi ) = 0.5*( 3. * l2( gi ) - 1. ) * (3. * l2( gi )   -2.) *  l2( gi )
       n( 3, gi ) = 0.5*( 3. * l3( gi ) - 1. ) * (3. * l3( gi )   -2.) *  l3( gi )
       ! mid side nodes...
       n( 4, gi ) = (9./2.)*l1( gi )*l2( gi )*( 3. * l1( gi ) - 1. )
       n( 5, gi ) = (9./2.)*l2( gi )*l1( gi )*( 3. * l2( gi ) - 1. )

       n( 6, gi ) = (9./2.)*l2( gi )*l3( gi )*( 3. * l2( gi ) - 1. )
       n( 7, gi ) = (9./2.)*l3( gi )*l2( gi )*( 3. * l3( gi ) - 1. )

       n( 8, gi ) = (9./2.)*l3( gi )*l1( gi )*( 3. * l3( gi ) - 1. )
       n( 9, gi ) = (9./2.)*l1( gi )*l3( gi )*( 3. * l1( gi ) - 1. )
       ! central node...
       n( 10, gi ) = 27.*l1( gi )*l2( gi )*l3( gi )

       ! x-derivative (nb. l1 + l2 + l3  = 1 )
       ! corner nodes...
       nlx( 1, gi ) = 0.5*( 27. * l1( gi )**2  - 18. *  l1( gi ) + 2. )
       nlx( 2, gi ) = 0.0
       nlx( 3, gi ) = 0.5*( 27. * l3( gi )**2  - 18. *  l3( gi ) + 2. )   *  (-1.0)
       ! mid side nodes...
       nlx( 4, gi ) = (9./2.)*(6.*l1( gi )*l2( gi )  - l2( gi ) )
       nlx( 5, gi ) = (9./2.)*l2( gi )*( 3. * l2( gi ) - 1. )

       nlx( 6, gi ) = - (9./2.)*l2( gi )*( 3. * l2( gi ) - 1. )
       nlx( 7, gi ) = (9./2.)*(   -l2(gi)*( 6.*l3(gi) -1. )    )

       nlx( 8, gi ) = -(9./2.)*( l1( gi )*(6.*l3(gi)-1.) + l3(gi)*(3.*l3(gi)-1.)  )
       nlx( 9, gi ) = (9./2.)*(  l3(gi)*(3.*l1(gi)-1.) -l1(gi)*(3.*l1(gi)-1.)  )
       ! central node...
       nlx( 10, gi ) = 27.*l2( gi )*( 1. - 2.*l1(gi)  - l2( gi ) )

       ! y-derivative (nb. l1 + l2 + l3  = 1 )
       ! corner nodes...
       nly( 1, gi ) = 0.0
       nly( 2, gi ) = 0.5*( 27. * l2( gi )**2  - 18. *  l2( gi ) + 2.  )
       nly( 3, gi ) = 0.5*( 27. * l3( gi )**2  - 18. *  l3( gi ) + 2.  )   *  (-1.0)
       ! mid side nodes...
       nly( 4, gi ) = (9./2.)*l1( gi )*( 3. * l1( gi ) - 1. )
       nly( 5, gi ) = (9./2.)*l1( gi )*( 6. * l2( gi ) - 1. )

       nly( 6, gi ) = (9./2.)*( l3( gi )*( 6. * l2( gi ) - 1. ) -l2(gi)*( 3.*l2(gi)-1. )  )
       nly( 7, gi ) = (9./2.)*( -l2( gi )*( 6. * l3( gi ) - 1. ) +l3(gi)*(3.*l3(gi)-1.)  )

       nly( 8, gi ) = -(9./2.)*l1( gi )*( 6. * l3( gi ) - 1. )
       nly( 9, gi ) = -(9./2.)*l1( gi )*( 3. * l1( gi ) - 1. )
       ! central node...
       nly( 10, gi ) = 27.*l1( gi )*( 1. - 2.*l2(gi)  - l1( gi ) )
    end do

  end subroutine shape_triangle_cubic


    	  
!
! 
       SUBROUTINE MATINV(A,N,NMAX,MAT,MAT2,X,B)
! This sub finds the inverse of the matrix A and puts it back in A. 
! MAT, MAT2 & X,B are working vectors. 
       IMPLICIT NONE
       INTEGER N,NMAX
       REAL A(NMAX,NMAX),MAT(N,N),MAT2(N,N),X(N),B(N)
! Local variables
       INTEGER ICOL,IM,JM


         DO IM=1,N
           DO JM=1,N
             MAT(IM,JM)=A(IM,JM)
           END DO
         END DO
! Solve MAT X=B (NB MAT is overwritten).  
       CALL SMLINN_FACTORIZE(MAT,X,B,N,N)
!
       DO ICOL=1,N
!
! Form column ICOL of the inverse. 
         DO IM=1,N
           B(IM)=0.
         END DO
         B(ICOL)=1.0
! Solve MAT X=B (NB MAT is overwritten).  
       CALL SMLINN_SOLVE_LU(MAT,X,B,N,N)
! X contains the column ICOL of inverse
         DO IM=1,N
           MAT2(IM,ICOL)=X(IM)
         END DO 
!
      END DO
!
! Set A to MAT2
         DO IM=1,N
           DO JM=1,N
             A(IM,JM)=MAT2(IM,JM)
           END DO
         END DO
       RETURN
       END SUBROUTINE MATINV
!
!
!     
	  
        SUBROUTINE SMLINN_FACTORIZE(A,X,B,NMX,N)
        IMPLICIT NONE
        INTEGER NMX,N
        REAL A(NMX,NMX),X(NMX),B(NMX)
        REAL R
        INTEGER K,I,J
!     Form X = A^{-1} B
!     Useful subroutine for inverse
!     This sub overwrites the matrix A. 
        DO K=1,N-1
           DO I=K+1,N
              A(I,K)=A(I,K)/A(K,K)
           END DO
           DO J=K+1,N
              DO I=K+1,N
                 A(I,J)=A(I,J) - A(I,K)*A(K,J)
              END DO
           END DO
        END DO
!     
      if(.false.) then
!     Solve L_1 x=b
        DO I=1,N
           R=0.
           DO J=1,I-1
              R=R+A(I,J)*X(J)
           END DO
           X(I)=B(I)-R
        END DO
!     
!     Solve U x=y
        DO I=N,1,-1
           R=0.
           DO J=I+1,N
              R=R+A(I,J)*X(J)
           END DO
           X(I)=(X(I)-R)/A(I,I)
        END DO
      endif
        RETURN
        END SUBROUTINE SMLINN_FACTORIZE
!     
!     
	  
        SUBROUTINE SMLINN_SOLVE_LU(A,X,B,NMX,N)
        IMPLICIT NONE
        INTEGER NMX,N
        REAL A(NMX,NMX),X(NMX),B(NMX)
        REAL R
        INTEGER K,I,J
!     Form X = A^{-1} B
!     Useful subroutine for inverse
!     This sub overwrites the matrix A. 
       if(.false.) then
        DO K=1,N-1
           DO I=K+1,N
              A(I,K)=A(I,K)/A(K,K)
           END DO
           DO J=K+1,N
              DO I=K+1,N
                 A(I,J)=A(I,J) - A(I,K)*A(K,J)
              END DO
           END DO
        END DO
       endif
!     
!     Solve L_1 x=b
        DO I=1,N
           R=0.
           DO J=1,I-1
              R=R+A(I,J)*X(J)
           END DO
           X(I)=B(I)-R
        END DO
!     
!     Solve U x=y
        DO I=N,1,-1
           R=0.
           DO J=I+1,N
              R=R+A(I,J)*X(J)
           END DO
           X(I)=(X(I)-R)/A(I,I)
        END DO
        RETURN
        END SUBROUTINE SMLINN_SOLVE_LU
!     
!       


!     
      SUBROUTINE JACOBL(N,ALPHA,BETA,XJAC)
      IMPLICIT NONE
!     COMPUTES THE GAUSS-LOBATTO COLLOCATION POINTS FOR JACOBI POLYNOMIALS
!     
!     N:       DEGREE OF APPROXIMATION
!     ALPHA:   PARAMETER IN JACOBI WEIGHT
!     BETA:    PARAMETER IN JACOBI WEIGHT
!     
!     XJAC:    OUTPUT ARRAY WITH THE GAUSS-LOBATTO ROOTS
!     THEY ARE ORDERED FROM LARGEST (+1.0) TO SMALLEST (-1.0)
!     
      INTEGER N
      REAL ALPHA,BETA
!      IMPLICIT REAL(A-H,O-Z)
!      REAL XJAC(1)
      REAL XJAC(N+1)
      REAL ALP,BET,RV
      REAL PNP1P,PDNP1P,PNP,PDNP,PNM1P,PDNM1,PNP1M,PDNP1M,PNM,PDNM,PNM1M
      REAL DET,RP,RM,A,B,DTH,CD,SD,CS,SS,X,PNP1,PDNP1,PN,PDN,PNM1,POLY
      REAL PDER,RECSUM,DELX,CSSAVE
      INTEGER NP,NH,J,K,JM,I,NPP
      COMMON /JACPAR/ALP,BET,RV
      INTEGER KSTOP
      DATA KSTOP/10/
      REAL EPS
      DATA EPS/1.0E-12/
      ALP = ALPHA
      BET =BETA
      RV = 1 + ALP
      NP = N+1
!
!  COMPUTE THE PARAMETERS IN THE POLYNOMIAL WHOSE ROOTS ARE DESIRED
!
      CALL JACOBF(NP,PNP1P,PDNP1P,PNP,PDNP,PNM1P,PDNM1,1.0)
      CALL JACOBF(NP,PNP1M,PDNP1M,PNM,PDNM,PNM1M,PDNM1,-1.0)
      DET = PNP*PNM1M-PNM*PNM1P
      RP = -PNP1P
      RM = -PNP1M
      A = (RP*PNM1M-RM*PNM1P)/DET
      B = (RM*PNP-RP*PNM)/DET
!
      XJAC(1) = 1.0
      NH = (N+1)/2
!
!  SET-UP RECURSION RELATION FOR INITIAL GUESS FOR THE ROOTS
!
      DTH = 3.14159265/(2*N+1)
      CD = COS(2.*DTH)
      SD = SIN(2.*DTH)
      CS = COS(DTH)
      SS = SIN(DTH)
!
!  COMPUTE THE FIRST HALF OF THE ROOTS BY POLYNOMIAL DEFLATION
!
      do  J=2,NH! Was loop 39
         X = CS
      do  K=1,KSTOP! Was loop 29
            CALL JACOBF(NP,PNP1,PDNP1,PN,PDN,PNM1,PDNM1,X)
            POLY = PNP1+A*PN+B*PNM1
            PDER = PDNP1+A*PDN+B*PDNM1
            RECSUM = 0.0
            JM = J-1
      do  I=1,JM! Was loop 27
               RECSUM = RECSUM+1.0/(X-XJAC(I))
      end do ! Was loop 27
28          CONTINUE
            DELX = -POLY/(PDER-RECSUM*POLY)
            X = X+DELX
            IF(ABS(DELX) .LT. EPS) GO TO 30
      end do ! Was loop 29
30       CONTINUE
         XJAC(J) = X
         CSSAVE = CS*CD-SS*SD
         SS = CS*SD+SS*CD
         CS = CSSAVE
      end do ! Was loop 39
      XJAC(NP) = -1.0
      NPP = N+2
!
! USE SYMMETRY FOR SECOND HALF OF THE ROOTS
!
      do  I=2,NH! Was loop 49
         XJAC(NPP-I) = -XJAC(I)
      end do ! Was loop 49
      IF(N .NE. 2*(N/2)) RETURN
      XJAC(NH+1) = 0.0
      RETURN
      END SUBROUTINE JACOBL


!     
!     
!     
      REAL FUNCTION LAGRAN(DIFF,LX,INOD,NDNOD,NODPOS)
      IMPLICIT NONE
!     This return the Lagrange poly assocaited with node INOD at point LX
!     If DIFF then send back the value of this poly differentiated. 
      LOGICAL DIFF
      INTEGER INOD,NDNOD
      REAL LX,NODPOS(0:NDNOD-1)
      REAL DENOMI,OVER,OVER1
      INTEGER N,K,I,JJ
!     ewrite(3,*) 'inside lagran'
!     ewrite(3,*) 'DIFF,LX,INOD,NDNOD,NODPOS:',
!     &            DIFF,LX,INOD,NDNOD,NODPOS
!
      N=NDNOD-1
      K=INOD-1
!     
      DENOMI=1.
      do I=0,K-1
         DENOMI=DENOMI*(NODPOS(K)-NODPOS(I))
      END DO
      do I=K+1,N
         DENOMI=DENOMI*(NODPOS(K)-NODPOS(I))
      END DO
!     
      IF(.NOT.DIFF) THEN
         OVER=1.
      do I=0,K-1
            OVER=OVER*(LX-NODPOS(I))
         END DO
      do I=K+1,N
            OVER=OVER*(LX-NODPOS(I))
         END DO
         LAGRAN=OVER/DENOMI
      ELSE
         OVER=0.
      do JJ=0,N
            IF(JJ.NE.K) THEN
               OVER1=1.
      do I=0,K-1
                  IF(JJ.NE.I) OVER1=OVER1*(LX-NODPOS(I))
               END DO
      do I=K+1,N
                  IF(JJ.NE.I) OVER1=OVER1*(LX-NODPOS(I))
               END DO
               OVER=OVER+OVER1
            ENDIF
         END DO
         LAGRAN=OVER/DENOMI
      ENDIF
!     
!     ewrite(3,*) 'FINISHED LAGRAN'
      END
      


      SUBROUTINE LAGROT(WEIT,QUAPOS,NDGI,GETNDP)
!        use RGPTWE_module
      IMPLICIT NONE
!     This computes the weight and points for standard Gaussian quadrature.
!     IF(GETNDP) then get the POSITION OF THE NODES 
!     AND DONT BOTHER WITH THE WEITS.
      INTEGER NDGI
      REAL WEIT(NDGI),QUAPOS(NDGI)
      LOGICAL GETNDP
      LOGICAL WEIGHT
      INTEGER IG
! real function...
      real RGPTWE
!     
      IF(.NOT.GETNDP) THEN
         WEIGHT=.TRUE.
         do IG=1,NDGI
            WEIT(IG)=RGPTWE(IG,NDGI,WEIGHT)
         END DO
!     
         WEIGHT=.FALSE.
         do IG=1,NDGI
            QUAPOS(IG)=RGPTWE(IG,NDGI,WEIGHT)
         END DO
      ELSE
         IF(NDGI.EQ.1) THEN
            QUAPOS(1)=0.
         ELSE
            do IG=1,NDGI
               QUAPOS(IG)= -1+2.*REAL(IG-1)/REAL(NDGI-1)
            END DO
         ENDIF
      ENDIF
      END SUBROUTINE LAGROT





  REAL FUNCTION RGPTWE(IG,ND,WEIGHT)
    IMPLICIT NONE
    !     NB If WEIGHT is TRUE in function RGPTWE then return the Gauss-pt weight 
    !     else return the Gauss-pt. 
    !     NB there are ND Gauss points we are looking for either the 
    !     weight or the x-coord of the IG'th Gauss point. 
    INTEGER IG,ND
    LOGICAL WEIGHT

    IF(WEIGHT) THEN
       if(ND==19) GO TO 19000
       GO TO (10,20,30,40,50,60,70,80,90,100) ND
       !     +++++++++++++++++++++++++++++++
       !     For N=1 +++++++++++++++++++++++
10     CONTINUE
       RGPTWE=2.0
       GO TO 1000
       !     For N=2 +++++++++++++++++++++++
20     CONTINUE
       RGPTWE=1.0
       GO TO 1000
       ! For N=3 +++++++++++++++++++++++
30     CONTINUE
       GO TO (11,12,11) IG
11     RGPTWE= 0.555555555555556
       GO TO 1000
12     RGPTWE= 0.888888888888889
       GO TO 1000
       ! For N=4 +++++++++++++++++++++++
40     CONTINUE
       GO TO (21,22,22,21) IG
21     RGPTWE= 0.347854845137454
       GO TO 1000
22     RGPTWE= 0.652145154862546
       GO TO 1000
       ! For N=5 +++++++++++++++++++++++
50     CONTINUE
       GO TO (31,32,33,32,31) IG
31     RGPTWE= 0.236926885056189
       GO TO 1000
32     RGPTWE= 0.478628670499366
       GO TO 1000
33     RGPTWE= 0.568888888888889
       GO TO 1000
       ! For N=6 +++++++++++++++++++++++
60     CONTINUE
       GO TO (41,42,43,43,42,41) IG
41     RGPTWE= 0.171324492379170
       GO TO 1000
42     RGPTWE= 0.360761573048139
       GO TO 1000
43     RGPTWE= 0.467913934572691
       GO TO 1000
       ! For N=7 +++++++++++++++++++++++
70     CONTINUE
       GO TO (51,52,53,54,53,52,51) IG
51     RGPTWE= 0.129484966168870
       GO TO 1000
52     RGPTWE= 0.279705391489277
       GO TO 1000
53     RGPTWE= 0.381830050505119
       GO TO 1000
54     RGPTWE= 0.417959183673469
       GO TO 1000
       ! For N=8 +++++++++++++++++++++++
80     CONTINUE
       GO TO (61,62,63,64,64,63,62,61) IG
61     RGPTWE= 0.101228536290376
       GO TO 1000
62     RGPTWE= 0.222381034453374
       GO TO 1000
63     RGPTWE= 0.313706645877877
       GO TO 1000
64     RGPTWE= 0.362683783378362
       GO TO 1000
       ! For N=9 +++++++++++++++++++++++
90     CONTINUE
       GO TO (71,72,73,74,75,74,73,72,71) IG
71     RGPTWE= 0.081274388361574
       GO TO 1000
72     RGPTWE= 0.180648160694857
       GO TO 1000
73     RGPTWE= 0.260610696402935
       GO TO 1000
74     RGPTWE= 0.312347077040003
       GO TO 1000
75     RGPTWE= 0.330239355001260
       GO TO 1000
       ! For N=10 +++++++++++++++++++++++
100    CONTINUE
       GO TO (81,82,83,84,85,85,84,83,82,81) IG
81     RGPTWE= 0.066671344308688
       GO TO 1000
82     RGPTWE= 0.149451349150581
       GO TO 1000
83     RGPTWE= 0.219086362515982
       GO TO 1000
84     RGPTWE= 0.269266719309996
       GO TO 1000
85     RGPTWE= 0.295524224714753
       GO TO 1000
       ! For N=19 +++++++++++++++++++++++
19000  CONTINUE
       GO TO (9991,9992,9993,9994,9995,9996,9997,9998,9999, 91000,  &
              9999,9998,9997,9996,9995,9994,9993,9992,9991) IG
9991     RGPTWE= 0.0194617882297265
       GO TO 1000
9992     RGPTWE= 0.0448142267656996
       GO TO 1000
9993     RGPTWE= 0.0690445427376412
       GO TO 1000
9994     RGPTWE= 0.0914900216224500
       GO TO 1000
9995     RGPTWE= 0.1115666455473340
       GO TO 1000
9996     RGPTWE= 0.1287539625393362
       GO TO 1000
9997     RGPTWE= 0.1426067021736066
       GO TO 1000
9998     RGPTWE= 0.1527660420658597
       GO TO 1000
9999     RGPTWE= 0.1589688433939543
       GO TO 1000
91000     RGPTWE= 0.1610544498487837
       GO TO 1000
! 
1000   CONTINUE
    ELSE
       IF(ND==19) GOTO 1090
       GO TO (210,220,230,240,250,260,270,280,290,200) ND
       ! +++++++++++++++++++++++++++++++
       ! For N=1 +++++++++++++++++++++++ THE GAUSS POINTS...
210    CONTINUE
       RGPTWE=0.0
       GO TO 2000
       ! For N=2 +++++++++++++++++++++++
220    CONTINUE
       RGPTWE= 0.577350269189626
       GO TO 2000
       ! For N=3 +++++++++++++++++++++++
230    CONTINUE
       GO TO (211,212,211) IG
211    RGPTWE= 0.774596669241483
       GO TO 2000
212    RGPTWE= 0.0
       GO TO 2000
       ! For N=4 +++++++++++++++++++++++
240    CONTINUE
       GO TO (221,222,222,221) IG
221    RGPTWE= 0.861136311594953
       GO TO 2000
222    RGPTWE= 0.339981043584856
       GO TO 2000
       ! For N=5 +++++++++++++++++++++++
250    CONTINUE
       GO TO (231,232,233,232,231) IG
231    RGPTWE= 0.906179845938664
       GO TO 2000
232    RGPTWE= 0.538469310105683
       GO TO 2000
233    RGPTWE= 0.0
       GO TO 2000
       ! For N=6 +++++++++++++++++++++++
260    CONTINUE
       GO TO (241,242,243,243,242,241) IG
241    RGPTWE= 0.932469514203152
       GO TO 2000
242    RGPTWE= 0.661209386466265
       GO TO 2000
243    RGPTWE= 0.238619186083197
       GO TO 2000
       ! For N=7 +++++++++++++++++++++++
270    CONTINUE
       GO TO (251,252,253,254,253,252,251) IG
251    RGPTWE= 0.949107912342759
       GO TO 2000
252    RGPTWE= 0.741531185599394
       GO TO 2000
253    RGPTWE= 0.405845151377397
       GO TO 2000
254    RGPTWE= 0.0
       GO TO 2000
       ! For N=8 +++++++++++++++++++++++
280    CONTINUE
       GO TO (261,262,263,264,264,263,262,261) IG
261    RGPTWE= 0.960289856497536
       GO TO 2000
262    RGPTWE= 0.796666477413627
       GO TO 2000
263    RGPTWE= 0.525532409916329
       GO TO 2000
264    RGPTWE= 0.183434642495650
       GO TO 2000
       ! For N=9 +++++++++++++++++++++++
290    CONTINUE
       GO TO (271,272,273,274,275,274,273,272,271) IG
271    RGPTWE= 0.968160239507626
       GO TO 2000
272    RGPTWE= 0.836031107326636
       GO TO 2000
273    RGPTWE= 0.613371432700590
       GO TO 2000
274    RGPTWE= 0.324253423403809
       GO TO 2000
275    RGPTWE= 0.0
       GO TO 2000
       ! For N=10 +++++++++++++++++++++++
200    CONTINUE
       GO TO (281,282,283,284,285,285,284,283,282,281) IG
281    RGPTWE= 0.973906528517172
       GO TO 2000
282    RGPTWE= 0.865063366688985
       GO TO 2000
283    RGPTWE= 0.679409568299024
       GO TO 2000
284    RGPTWE= 0.433395394129247
       GO TO 2000
285    RGPTWE= 0.148874338981631
       GO TO 2000
       !
       ! For N=19 +++++++++++++++++++++++
1090    CONTINUE
       GO TO (181,182,183,184,185,186,187,188,189, 190, &
              189,188,187,186,185,184,183,182,181) IG
181    RGPTWE= 0.9924068438435844
       GO TO 2000
182    RGPTWE= 0.9602081521348300
       GO TO 2000
183    RGPTWE= 0.9031559036148179
       GO TO 2000
184    RGPTWE= 0.8227146565371428
       GO TO 2000
185    RGPTWE= 0.7209661773352294
       GO TO 2000
186    RGPTWE= 0.6005453046616810
       GO TO 2000
187    RGPTWE= 0.4645707413759609
       GO TO 2000
188    RGPTWE= 0.3165640999636298
       GO TO 2000
189    RGPTWE= 0.1603586456402254
       GO TO 2000
190    RGPTWE= 0.0
       GO TO 2000
! 
2000   CONTINUE
       IF(IG.LE.INT((ND/2)+0.1)) RGPTWE=-RGPTWE
    ENDIF
! the web link provides extensive list of 
! https://pomax.github.io/bezierinfo/legendre-gauss.html
  END FUNCTION RGPTWE





      SUBROUTINE JACOBF(N,POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,X)
      IMPLICIT NONE
!     
!     COMPUTES THE JACOBI POLYNOMIAL (POLY) AND ITS DERIVATIVE
!     (PDER) OF DEGREE  N  AT  X
!     
      INTEGER N
      REAL APB,POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,X
!     IMPLICIT REAL(A-H,O-Z)
      COMMON /JACPAR/ALP,BET,RV
      REAL ALP,BET,RV,POLYLST,PDERLST,A1,A2,B3,A3,A4
      REAL POLYN,PDERN,PSAVE,PDSAVE
      INTEGER K
      APB = ALP+BET
      POLY = 1.0
      PDER = 0.0
      IF(N .EQ. 0) RETURN
      POLYLST = POLY
      PDERLST = PDER
      POLY = RV * X
      PDER = RV
      IF(N .EQ. 1) RETURN
      do K=2,N
         A1 = 2.*K*(K+APB)*(2.*K+APB-2.)
         A2 = (2.*K+APB-1.)*(ALP**2-BET**2)
         B3 = (2.*K+APB-2.)
         A3 = B3*(B3+1.)*(B3+2.)
         A4 = 2.*(K+ALP-1)*(K+BET-1.)*(2.*K+APB)
         POLYN = ((A2+A3*X)*POLY-A4*POLYLST)*A1
         PDERN = ((A2+A3*X)*PDER-A4*PDERLST+A3*POLY)*A1
         PSAVE = POLYLST
         PDSAVE = PDERLST
         POLYLST = POLY
         POLY = POLYN
         PDERLST = PDER
         PDER = PDERN
      END DO
      POLYM1 = POLYLST
      PDERM1 = PDERLST
      POLYM2 = PSAVE
      PDERM2 = PDSAVE
      RETURN
      END SUBROUTINE JACOBF




 
