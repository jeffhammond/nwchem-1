      subroutine ccsd_trpdrv_omp_fbody(
     &     f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,eorb,
     &     ncor,nocc,nvir, emp4,emp5,
     &     a,i,j,k,klo,
     &     Tij, Tkj, Tia, Tka, Xia, Xka,
     &     Jia, Jka, Kia, Kka,
     &     Jij, Jkj, Kij, Kkj,
     &     dintc1,dintx1,t1v1,
     &     dintc2,dintx2,t1v2)
      implicit none
      double precision emp4,emp5
      integer ncor,nocc,nvir
      integer a,i,j,k, klo
      double precision f1n(nvir,nvir),f1t(nvir,nvir)
      double precision f2n(nvir,nvir),f2t(nvir,nvir)
      double precision f3n(nvir,nvir),f3t(nvir,nvir)
      double precision f4n(nvir,nvir),f4t(nvir,nvir)
      double precision eorb(*)
      double precision Tij(*), Tkj(*), Tia(*), Tka(*)
      double precision Xia(*), Xka(*)
      double precision Jia(*), Jka(*), Jij(*), Jkj(*)
      double precision Kia(*), Kka(*), Kij(*), Kkj(*)
      double precision dintc1(nvir),dintx1(nvir)
      double precision dintc2(nvir),dintx2(nvir)
      double precision t1v1(nvir),t1v2(nvir)
      double precision emp4i,emp5i,emp4k,emp5k
      double precision eaijk,denom
      integer lnov,lnvv
      integer b,c

      lnov=nocc*nvir
      lnvv=nvir*nvir

      emp4i = 0.0d0
      emp5i = 0.0d0
      emp4k = 0.0d0
      emp5k = 0.0d0

      eaijk = eorb(a) - ( eorb(ncor+i)+eorb(ncor+j)+eorb(ncor+k) )

!$omp target data map(to: f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t )
!$omp&            map(to: dintc1, dintc2, dintx1, dintx2, t1v1, t1v2 )
!$omp&            map(to: eorb, ncor, nocc, nvir, eaijk)
!$omp&            map(tofrom: emp5i, emp4i, emp5k, emp4k)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Jia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f1n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tia,nvir,Kkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f1n,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Kia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f2n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xia,nvir,Kkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f2n,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Jia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f3n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tia,nvir,Jkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f3n,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Kia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f4n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xia,nvir,Jkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f4n,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Jka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f1t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tka(1+(k-klo)*lnov),nvir,Kij,nocc,1.0d0,
     2     f1t,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Kka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f2t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xka(1+(k-klo)*lnov),nvir,Kij,nocc,1.0d0,
     2     f2t,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Jka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f3t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tka(1+(k-klo)*lnov),nvir,Jij,nocc,1.0d0,
     2     f3t,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Kka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f4t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xka(1+(k-klo)*lnov),nvir,Jij,nocc,1.0d0,
     2     f4t,nvir)


!$omp target
!$omp parallel do collapse(2)
!$omp& schedule(static)
!$omp& shared(eorb,eaijk)
!$omp& shared(f1n,f2n,f3n,f4n,f1t,f2t,f3t,f4t)
!$omp& shared(t1v1,dintc1,dintx1)
!$omp& shared(t1v2,dintc2,dintx2)
!$omp& private(denom)
!$omp& firstprivate(ncor,nocc,nvir)
!$omp& reduction(+:emp5i,emp4i)
!$omp& reduction(+:emp5k,emp4k)
      do b=1,nvir
        do c=1,nvir
          denom=-1.0d0/(eorb(ncor+nocc+b)+eorb(ncor+nocc+c)+eaijk)
          emp4i=emp4i+denom*
     &         (f1t(b,c)+f1n(c,b)+f2t(c,b)+f3n(b,c)+f4n(c,b))*
     &         (f1t(b,c)-2*f2t(b,c)-2*f3t(b,c)+f4t(b,c))
     &               -denom*
     &         (f1n(b,c)+f1t(c,b)+f2n(c,b)+f3n(c,b))*
     &         (2*f1t(b,c)-f2t(b,c)-f3t(b,c)+2*f4t(b,c))
     &               +3*denom*(
     &         f1n(b,c)*(f1n(b,c)+f3n(c,b)+2*f4t(c,b))+
     &         f2n(b,c)*f2t(c,b)+f3n(b,c)*f4t(b,c))
          emp4k=emp4k+denom*
     &         (f1n(b,c)+f1t(c,b)+f2n(c,b)+f3t(b,c)+f4t(c,b))*
     &         (f1n(b,c)-2*f2n(b,c)-2*f3n(b,c)+f4n(b,c))
     &               -denom*
     &         (f1t(b,c)+f1n(c,b)+f2t(c,b)+f3t(c,b))*
     &         (2*f1n(b,c)-f2n(b,c)-f3n(b,c)+2*f4n(b,c))
     &               +3*denom*(
     &         f1t(b,c)*(f1t(b,c)+f3t(c,b)+2*f4n(c,b))+
     &         f2t(b,c)*f2n(c,b)+f3t(b,c)*f4n(b,c))
          emp5i=emp5i+denom*t1v1(b)*dintx1(c)*
     &        (    f1t(b,c)+f2n(b,c)+f4n(c,b)
     &         -2*(f3t(b,c)+f4n(b,c)+f2n(c,b)+
     &             f1n(b,c)+f2t(b,c)+f3n(c,b))
     &         +4*(f3n(b,c)+f4t(b,c)+f1n(c,b)))
     &               +denom*t1v1(b)*dintc1(c)*
     &        (     f1n(b,c)+f4n(b,c)+f1t(c,b)
     &          -2*(f2n(b,c)+f3n(b,c)+f2t(c,b)))
          emp5k=emp5k+denom*t1v2(b)*dintx2(c)*
     &        (    f1n(b,c)+f2t(b,c)+f4t(c,b)
     &         -2*(f3n(b,c)+f4t(b,c)+f2t(c,b)+
     &             f1t(b,c)+f2n(b,c)+f3t(c,b))
     &         +4*(f3t(b,c)+f4n(b,c)+f1t(c,b)))
     &               +denom*t1v2(b)*dintc2(c)*
     &        (     f1t(b,c)+f4t(b,c)+f1n(c,b)
     &          -2*(f2t(b,c)+f3t(b,c)+f2n(c,b)))
        enddo
      enddo
!$omp end parallel do
!$omp end target
!$omp end target data

      emp4 = emp4 + emp4i
      emp5 = emp5 + emp5i
      if (i.ne.k) then
          emp4 = emp4 + emp4k
          emp5 = emp5 + emp5k
      end if ! (i.ne.k)
      end
