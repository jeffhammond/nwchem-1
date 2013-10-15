*
* On the storage of CI-coefficients in core
* =========================================
* I am currently reactivating the ICISTR = 1 option where two 
* complete vetors are in core. The purpose is to prepare for the 
* reintroduction of CSF's and MCSCF for small CI expansions. 
* However, the Sigma-generation and other matters should still be done in 
* batches defined by LCSBLK....!
* So the memory requirements for CI will be
* ICISTR = 1: Two vectors over all variables and three blocks
* IF CSF's are in use, then three vectors over SD's are also allocated at the moment
*
* A note on integrals
* ===================
*
* =========================================================
*. Standard (old) ordering, corresponding to ITRA_ROUTE = 1
* =========================================================
*
* The integrals are stored in LUCIA in MOLCAS order. The integrals (ij!kl)
* are thus organized in symmetryblocks with ism > jsm, ksm > lsm, 
* ijsm > klsm. Within a given block, the ordering goes as follows.
*
* i,j => ij: rowwise ordered  with i geq j if ism = jsm
* The index ij is thus defined as
* ij = i*(i-1)/2 + j (ism = jsm)
* ij = (i-1)*nj + j  (ism neq jsm)
*
* k,l => kl: rowwise ordered with k geq l if ksm = lsm
* The index kl is thus defined as
* kl = k*(k-1)/2 + l (ksm = lsm)
* kl = (k-1)*nl + l  (ksm neq lsm)
*
* ij, kl => ijkl: columnwise ordered with ij geq kl if ijsm = klsm
* The index ijkl is thus defined as 
* ijkl = (kl-1)*nij + ij - kl*(kl-1)/2 (ijsm = klsm)
* ijkl = (kl-1)*nij + ij               (ijsm neq klsm)
*
*
* =========================================================
*. New ordering, corresponding to ITRA_ROUTE = 2
* =========================================================
*
* Well, the LUCIA development team (aka Jeppe) find it inconvenient 
* to have the row-ordering for indeces IJ for ism. ne. jsm. 
* This is convenient for working with general integral arrays
* ( I hope and assume...). The only difference is therefore
* the mapping from pair of orbital indices p, q to compound pair index pq
*
* With respect to the symmetry blocks of the integral, these blocks 
* are still  organized in symmetryblocks with ism > jsm, ksm > lsm, 
* ijsm > klsm. Within a given block, the ordering goes as follows.
*
* i,j => ij: 
* ij = i*(i-1)/2 + j (ism = jsm)
* ij = (j-1)*ni + i  (ism neq jsm) ( ie now column ordered)
*
* k,l => kl: 
* kl = k*(k-1)/2 + l (ksm = lsm)
* kl = (l-1)*nk + k  (ksm neq lsm)
*
* ij, kl => ijkl: columnwise ordered with ij geq kl if ijsm = klsm (still)
* The index ijkl is thus defined as 
* ijkl = (kl-1)*nij + ij - kl*(kl-1)/2 (ijsm = klsm)
* ijkl = (kl-1)*nij + ij               (ijsm neq klsm)
*
*
* June 2010: About core energy: It has been decided at a recent
*            board meeting for LUCIA, that all routines per se
*            should calculate the energies including core-energies-
*            it is to messy with a number of different definitions..
*            This has been adapted for the standard CI path,
*            but must be added a number of other placed
* June 2010: Explicit introduction of inactive and secondary
*            orbitals. 
*. These orbitalsspaces have previously been handled as 
*. GAS-spaces, but I need the explicit treatment for efficiency.
*. So
*  1) Strings: start with orbital NINOB + 1
*  2) Integrals: Indeces are all over all orbitals
*  3) Density matrices: Indeces are over only active orbitals 
*  Inactive orbitals have type/orbital subspace 0 and 
*  secondary orbitals hav type/orbital subpaces NGAS + 1
*
*. A note on one-electron integrals
* KINT1: Integrals to be used in CI/energy..., may be of any type
*        is not used for permanent storage of any integrals, but
*        integrals are copied to this adress before use
*        defined before each call to e.g. direct CI
* KH  : One-electron integrals in MO basis
* KHINA: One-electron integrals with contributions from inactive
* orbitals- but not from particle-hole reorganization 
* KFI: One-electron integrals with contributions from inactive
* orbitals and from particle-hole reorganization 
*
*. Core-energies:
*  ECORE_EXT: Core-energy, usually from nuclear-nuclear repulsion energy
*  ECORE_INA: ECORE_EXT + contributions from one-and two-electron
*  operators for inactive orbitals
*  ECORE_FI: ECORE_INA + contributions from one- and two-electron
*  contributions from particle-hole reorganization
*
* Two-electron integrals 
* KINT2: Pointer to two-electron integrals 'in action'. 
* KINT_2EMO: Two-electron integrals in MO basis
* KINT_2EINI:Two-electron integrals in initial MO/AO basis- not always
*            defined or used
* 
* KMOAO: Points to which set of MO expansion coefficents, say MOAOIN
*        or MOAOUT, that currently are in use
*
* The MOAO expansion in use is also stored in KMOAO_ACT
* A set of transformed two-electron integrals are defined by two indices
*
* I2INT_NGENIND: Number of general indices in 2-electron integrals
* I2INT_OCINC  =1: Occupied is only active
*              =2: Occupied is active and inactive
*
* The orbitals are thus classified as occupied (O) or  general(G).
* A general orbital may be occupied. The occupied may either be 
* the active or the active + inactive orbitals.
*
* The following form of transformed integral lists may then be 
* obtained:
*
* I2INT_NGENIND = 0: (OO!OO)
*               = 1: (OO!OG) 
*               = 2: (OG!OG), (OO!GG)
*               = 3: (OG!GG)
*               = 4: (GG!GG)
* There is in general some redundancies in the above integral lists.
* First of all, no permutational symmetry is assumed between the general
* and occupied indeces, so for example in the (OO!OO) part of the (OO!OG)
* integrals, there is no use of the existing permutational symmetry
* between indices 3 and 4. Furthermore,in the integral lists (OG!OG), (OO!GG)
* integrals (OO!OG) occur a total of four times. However, these redundancies
* do not effect leading order terms of flop counts or memory.

*
* Route after call to MV7:
* MV7 -- RASSG3 -- SBLOCK -- SBLOCKS --- RSSBCB2 -- RSBB1E
*                                                -- RSBB2A
*                                                -- RSBB2BN2
* -----------------------------------------------------------
*
* Combined with QDOT code again, Febr. 2003 

* Note pt. CC calculations can be restarted from CI
* calculations in the same space by specifying CI=>CC. 
* This requires that the input CI vector and the CC 
* vector is in the same space. It would be 
* better to do the reformatting after the CI. 
*
* One can then do f.ex a partial CISDTQ to initialize the 
* CCSD.
*
* It would be 
*
* Lucia.f: GAS implementing no pair relativistic Theory
*
* Version of Febr 2003, Jeppe Olsen
* 
      SUBROUTINE GET_CMOAO_FUSK(CMO,NMOS_ENV)
*
* Fusk routine, setting CMO = 1
*
* Jeppe Olsen, May 2003
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*. Input
      DIMENSION NMOS_ENV(NIRREP)
*. Output
      DIMENSION CMO(*)
*
      IB = 1
      DO IRREP = 1, NIRREP
        IF(IRREP.EQ.1) THEN
          IB = 1
        ELSE 
          IB = IB + NMOS_ENV(IRREP-1)**2
        END IF
*
        LEN = NMOS_ENV(IRREP)**2
        ZERO = 0.0D0
        CALL SETVEC(CMO(IB),ZERO,LEN)
        ONE = 1.0D0
        CALL SETDIA(CMO(IB),ONE,NMOS_ENV(IRREP),0)
      END DO
*
      RETURN
      END

      SUBROUTINE GET_CMOAO_QDOT(CMO,NMOS_ENV,NAOS_ENV)
*
* Obtain MO-AO tranformation matrix form QDOT environment
*
      INCLUDE 'implicit.inc'
*. Input: Total number of orbitals and AO's, it is 
*          assumed that the number of electron and hole orbitals 
*          are identical 
      INTEGER NMOS_ENV(*), NAOS_ENV(*)
*. output 
      DIMENSION CMO(*)
*
* . Open 
      LUCVEC = IGETUNIT(71)
      OPEN(UNIT=LUCVEC,FILE='FINAL_CMAT',STATUS='UNKNOWN')
*. number of symmetries and number of systems
      READ(LUCVEC,*)  NSYM_ENV, NSYS_ENV
*. Skip number of orbitals per symmetry and system
      DO ISYS = 1, NSYS_ENV
       DO ISM = 1, NSYM_ENV
        READ(LUCVEC,*)
       END DO
      END DO
*.
      IOFF=1
      DO ISYS = 1,NSYS_ENV
       DO ISYM=1,NSYM_ENV
        LEN = NMOS_ENV(ISYM)/2
        CALL READ_CMAT(CMO(IOFF),LEN,LEN,LUCVEC)
        IOFF = IOFF + LEN*LEN                      
       END DO
      END DO
*
      CLOSE(UNIT=LUCVEC)
*
      RETURN
      END 
      SUBROUTINE READ_CMAT(C,NDIM,NMXDIM,LUCVEC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(NMXDIM,NMXDIM)
       
      DO I=1,NDIM
       DO J=1,NDIM
        READ(LUCVEC,*) C(J,I)
       END DO
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
       WRITE(6,*) ' Block of CMOAO read in '
       CALL WRTMAT(C,NDIM,NDIM,NDIM,NDIM)
      END IF
*
      RETURN
      END
      SUBROUTINE KERNEL_ROU_STAT_PRINT
*
* Print info on the use of kernel subroutines
*
      INCLUDE 'implicit.inc'
      INCLUDE 'rou_stat.inc'
*
      WRITE(6,*) ' ========'
      WRITE(6,*) '  MATCG: '
      WRITE(6,*) ' ========'
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_MATCG
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_MATCG
      WRITE(6,*)
      WRITE(6,*) ' ========'
      WRITE(6,*) '  MATCAS: '
      WRITE(6,*) ' ========'
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_MATCAS
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_MATCAS
      WRITE(6,*)
      WRITE(6,*) ' ============'
      WRITE(6,*) '  ADD_SKAIIB: '
      WRITE(6,*) ' ============'
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_ADD_SKAIIB
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_ADD_SKAIIB
      WRITE(6,*)
      WRITE(6,*) ' ============'
      WRITE(6,*) '  GET_CKAJJB: '
      WRITE(6,*) ' ============'
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_GET_CKAJJB
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_GET_CKAJJB
      WRITE(6,*)
      WRITE(6,*) ' =========='
      WRITE(6,*) '  COPVEC: '
      WRITE(6,*) ' =========='
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_COPVEC
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_COPVEC
      WRITE(6,*)
      WRITE(6,*) ' =========='
      WRITE(6,*) '  SETVEC: '
      WRITE(6,*) ' =========='
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_SETVEC
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_SETVEC
      WRITE(6,*)
      WRITE(6,*) ' =========='
      WRITE(6,*) '  SCALVE: '
      WRITE(6,*) ' =========='
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_SCALVE
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_SCALVE
      WRITE(6,*)
      WRITE(6,*) ' ========'
      WRITE(6,*) '  TRPMT: '
      WRITE(6,*) ' ========'
      WRITE(6,*) 
      WRITE(6,*) 'Number of calls = ', NCALL_TRPMT
      WRITE(6,'(A,E8.3)') ' Number of operations = ', XOP_TRPMT
*
      WRITE(6,*)
      WRITE(6,*) ' ======================= '
      WRITE(6,*) ' I/O traffic (R*8) words '
      WRITE(6,*) ' ======================= '
      WRITE(6,*)
      WRITE(6,'(A,E8.3)') ' Number of words written by TODSCP  =', 
     &                    XOP_TODSCP
      WRITE(6,'(A,E8.3)') ' Number of words written by TODSC   =', 
     &                    XOP_TODSC
      WRITE(6,'(A)') '----------------------------------------------'
      WRITE(6,'(A,E8.3)') ' Number of words written by TODSC*  =', 
     &              XOP_TODSCP+XOP_TODSC
      WRITE(6,'(A)') '----------------------------------------------'
      WRITE(6,*)
      WRITE(6,'(A,E8.3)') ' Number of words read by FRMDSC   =', 
     &                    XOP_FRMDSC
      WRITE(6,'(A,E8.3)') ' Number of words read by FRMDSC2  =', 
     &                    XOP_FRMDSC2
      WRITE(6,'(A,E8.3)') ' Number of words read by FRMDSCE  =', 
     &                    XOP_FRMDSCE
      WRITE(6,'(A,E8.3)') ' Number of words read by FRMDSCO  =', 
     &                    XOP_FRMDSCO
      WRITE(6,'(A)') '----------------------------------------------'
      WRITE(6,'(A,E8.3)') ' Number of words read by FRMDSC*  =',
     & XOP_FRMDSC+ XOP_FRMDSC2+XOP_FRMDSCE+XOP_FRMDSCO
      WRITE(6,'(A)') '----------------------------------------------'

      RETURN
      END
      SUBROUTINE KERNEL_ROU_STAT_INI
*
* Initialize information about the use of various 
* kernel subroutines ( In addition to matml )
*
      INCLUDE 'implicit.inc'
      INCLUDE 'rou_stat.inc'
C     COMMON/ROU_STAT/NCALL_SCALVE,NCALL_SETVEC,NCALL_COPVEC,
C    &                NCALL_MATCG,NCALL_MATCAS,NCALL_ADD_SKAIIB,
C    &                NCALL_GET_CKAJJB,
C    &                XOP_SCALVE,XOP_SETVEC,XOP_COPVEC,
C    &                XOP_MATCG,XOP_MATCAS,XOP_ADD_SKAIIB,
C    &                XOP_GET_CKAJJB
*
      NCALL_SCALVE = 0
      NCALL_SETVEC = 0
      NCALL_COPVEC = 0
      NCALL_MATCG = 0
      NCALL_MATCAS = 0
      NCALL_ADD_SKAIIB = 0
      NCALL_GET_CKAJJB = 0
      NCALL_TRPMT = 0
*
      XOP_SCALVE = 0
      XOP_SETVEC = 0
      XOP_COPVEC = 0
      XOP_MATCG = 0
      XOP_MATCAS = 0
      XOP_ADD_SKAIIB = 0
      XOP_GET_CKAJJB = 0
      XOP_TRPMT = 0
*. For I/O
      XOP_TODSCP = 0
      XOP_TODSC = 0
*
      XOP_FRMDSC  = 0
      XOP_FRMDSC2 = 0
      XOP_FRMDSCE = 0
      XOP_FRMDSCO = 0


      RETURN
      END 
      SUBROUTINE GET_E0(E0,EREF)
*
* Obtain E0 
*
*. Jeppe Olsen, November 1999
*
c      INCLUDE 'implicit.inc'
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cecore.inc'
*. 
      IDUMMMY = 1
      CALL MEMMAN(IDUMMY,IDUMMY,'MARK  ', IDUMMY,'GET_E0')
*
      IF(IE0AVEX.EQ.1) THEN
*. Obtain <0!f|0>
        CALL COPVEC(WORK(KINT1O),WORK(KFI),NINT1)
        CALL FIFAM(WORK(KFI))
        CALL SWAPVE(WORK(KFI),WORK(KINT1),NINT1)
        CALL EN_FROM_DENS(E0,1,0)
        CALL SWAPVE(WORK(KFI),WORK(KINT1),NINT1)
      ELSE IF(IE0AVEX.EQ.2) THEN
        E0 = EREF
      ELSE IF(IE0AVEX.EQ.3) THEN
        E0 = E0READ
      END IF
*. Core energy should not be included so
      E0 = E0 - ECORE
*
      NTEST = 100
      IF(NTEST.GE.100) THEN 
        WRITE(6,*) ' GET_E0, IE0AVEX, E0', IE0AVEX,E0 
      END IF
*
      RETURN
      END
      SUBROUTINE CHK_ORBDIM(IGSFILL,ISECFILL)
*
* Insert dimensions of orbital space IGSFILL or Secondary space
* Check number of shells in NGSSH with info from ENVIRONMENT
*
*. Determine also total number of inactive, active, and secondary orbitals
* 
* Environment info must be available
*
* Jeppe Olsen, Feb. 1998
*
*. Last modification; Jeppe Olsen; July 2013; Calc total dimensions of orbitals
*.                                            assuming shells = orbitals...
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
*
      NTEST = 100
*
      IF(IGSFILL.NE.0) THEN
*. Fill GAS shell IGSFILL with remaining orbitals
       DO IRREP = 1, NIRREP 
         LMO = NINASH(IRREP) + NSECSH(IRREP)
         DO IGAS = 1, NGAS
           IF(IGAS.NE.IGSFILL) LMO = LMO + NGSSH(IRREP,IGAS)
         END DO
         NGSSH(IRREP,IGSFILL)  = NMOS_ENV(IRREP)- LMO 
       END DO
      ELSE IF(ISECFILL.NE.0) THEN
        DO IRREP = 1, NIRREP
         NORBS = NINASH(IRREP)
         DO IGAS = 1, NGAS
          NORBS = NORBS + NGSSH(IRREP,IGAS)
         END DO
         NSECSH(IRREP) =  NMOS_ENV(IRREP) - NORBS 
        END DO
      END IF
*. Make sure that no dimensions are negative
      LERROR = 0
      DO IGAS = 1, NGAS
       DO IRREP = 1, NIRREP
         IF(NGSSH(IGAS,IRREP).LT.0) THEN
           WRITE(6,*) 
     &     ' Error: negative orbital dimension,IGAS,IRREP,N ',
     &     IGAS,IRREP,NGSSH(IGAS,IRREP) 
           LERROR = LERROR + 1
         END IF
       END DO
      END DO
*. And the secondary space
      DO IRREP = 1, NIRREP
        IF(NSECSH(IRREP).LT.0) THEN
           WRITE(6,*) 
     &     ' Error: negative orbital dimension for sec space,IRREP,N ',
     &     IRREP,NSECSH(IRREP) 
           LERROR = LERROR + 1
         END IF
      END DO
*. Make sure that all dimensions add correctly up
      DO IRREP = 1, NIRREP
        LMO = 0
        DO IGAS = 1, NGAS
          LMO =   LMO + NGSSH(IRREP,IGAS) 
        END DO
        LMO = LMO + NINASH(IRREP) + NSECSH(IRREP)
        IF(LMO.NE.NMOS_ENV(IRREP)) THEN
          WRITE(6,*) 
     &    ' Error: Number of orbitals in irrep not consistent'
          WRITE(6,*)
     &    ' with information from environment, IRREP,NMO,NMO_ENV' 
          WRITE(6,'(3I5)') IRREP,LMO,NMOS_ENV(IRREP)
          LERROR = LERROR + 1
        END IF
      END DO
*
      IF(LERROR.NE.0) THEN
        WRITE(6,*) ' Problem with orbital dimensions'
        STOP       ' Problem with orbital dimensions'
      END IF
*
      NINOB = IELSUM(NINASH,NIRREP)
      NSCOB = IELSUM(NSECSH,NIRREP)
      NACOB = 0
      DO IGAS = 1, NGAS
        NACOB = NACOB + IELSUM(NGSSH(1,IGAS),NIRREP)
      END DO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' CHK_ORBDIM:  NINOB, NACOB, NSCOB = ',
     &                            NINOB, NACOB, NSCOB
      END IF
*
      RETURN
      END
      SUBROUTINE GET_CMOAO_ENV(CMO)
*
* Obtain AO-MO transformation matrix from Environment
*
* Jeppe Olsen, November 1997
*              QDOT added (again), Feb. 2003
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'orbinp.inc'
*. Output
      DIMENSION CMO(*)
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*)
        WRITE(6,*) ' Info from GET_CMOAO_ENV '
        WRITE(6,*) ' ========================'
        WRITE(6,*)
        WRITE(6,*) ' ENVIRO = ', ENVIRO
      END IF

      IF(ENVIRO(1:6).EQ.'DALTON') THEN
        CALL GET_CMOAO_DALTON(CMO,NMOS_ENV(1),NAOS_ENV(1),NSMOB)
      ELSE IF(ENVIRO(1:6).EQ.'MOLCAS') THEN
*. Readin from LUMORB file
        CALL GETMOAO_MOLCAS(CMO,LUMOIN)
      ELSE IF(ENVIRO(1:5).EQ.'LUCIA' ) THEN
*. Read in from LUCIA 1e file: unit 91
        LU91 = 91
        CALL GET_CMOAO_LUCIA(CMO,NMOS_ENV,NAOS_ENV,LU91)
      ELSE IF (ENVIRO(1:4).EQ.'QDOT') THEN
*. Obtain CMOAO coefficients from file FINAL_CMAT
        CALL GET_CMOAO_QDOT(CMO,NMOS_ENV,NAOS_ENV)
      ELSE IF(ENVIRO(1:4).EQ.'NONE') THEN
        WRITE(6,*) ' GET_CMOAO, Warning: Called with ENVIRO = NONE'
        WRITE(6,*) ' No coefficients read in '
      ELSE IF(ENVIRO(1:4).EQ.'FUSK') THEN
*. Set the CMOAO matrices to be unit matrices, NMOS_ENV = NAOS_ENV is asssumed
        CALL GET_CMOAO_FUSK(CMO,NMOS_ENV)
      END IF
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' CMOAO matrix from GET_CMOAO_ENV'
        CALL APRBLM2(CMO,NMOS_ENV,NMOS_ENV,NSMOB,0)
      END IF
*
      RETURN
      END
      SUBROUTINE GETOBS_DALTON(ECORE_ENV,NAOS_ENV,NMOS_ENV)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CHARACTER*72 TITMOL(2)
      CHARACTER*8 LABEL8
*. Scratch
      DIMENSION TITLE(24),NBAS(8), NOCC(8), NLAMDA(8), NORB(8)
*. Output
      DIMENSION NAOS_ENV(*), NMOS_ENV(*)

*
* AO info
*
C     Read information on file AONEINT from HERMIT.
      ITAP34 = 66
      OPEN (ITAP34,STATUS='OLD',FORM='UNFORMATTED',FILE='AOONEINT')
      REWIND ITAP34
*. It seems like AOONEINT has changed so(Jan 11)
      INEW_OR_OLD = 1
      IF(INEW_OR_OLD.EQ.2) THEN
        READ (ITAP34) TITLE,NST,(NBAS(I),I=1,NST),ENUC
      ELSE
        READ(ITAP34) TITMOL
        READ (ITAP34) NST,(NBAS(I),I=1,NST),ENUC
      END IF
      CLOSE(ITAP34,STATUS='KEEP')
      ECORE_ENV = ENUC
C     WRITE(6,'(//A,2(/12A6)/)')
C    *   ' Dalton   title from basis set input:',(TITLE(I),I=1,24)
*
C     WRITE(6,*) ' Number of basis functions per sym '
C     CALL IWRTMA(NBAS,NST,1,NST,1)
*
      CALL ICOPVE(NBAS,NAOS_ENV,NST)
C
C     Read information on file SIRIFC written from SIRIUS.
C
*
* MO info
*
*. By trial and error - EKD + JO, NLAMDA was identified as 
*. the array holding number of MO's
*
      ITAP30 = 16
      OPEN(ITAP30,STATUS='OLD',FORM='UNFORMATTED',FILE='SIRIFC')
      REWIND ITAP30
C?    WRITE(6,*) ' GETOBS_DALTON BEFORE MOLLAB, TRCCINT'
      LABEL8='TRCCINT '
      LUERR = 6
      CALL MOLLAB(LABEL8,ITAP30,LUERR)
C?    WRITE(6,*) ' GETOBS_DALTON AFTER MOLLAB, TRCCINT'
      READ (ITAP30) NSYMHF,NORBT,NBAST,NCMOT,(NOCC(I),I=1,NSYMHF), 
     *              (NLAMDA(I),I=1,NSYMHF),(NORB(I),I=1,NSYMHF),
     *              POTNUC,EMCSCF
      CALL ICOPVE(NLAMDA,NMOS_ENV,NST)
C?    WRITE(6,*) ' Norb as delivered from environment '
C?    CALL IWRTMA(NORB,1,8,1,8)
*
C?    WRITE(6,*) ' NOCC NLAMDA  as delivered from DALTON'
C?    CALL IWRTMA(NOCC,1,8,1,8)
C?    CALL IWRTMA(NLAMDA,1,8,1,8)
*. 
C?    WRITE(6,*) ' NORBT, NCMOT = ', NORBT,NCMOT
      RETURN
      END
      SUBROUTINE GET_ORB_DIM_ENV(ECORE_ENV)
*
* Obtain number of orbitals and basis functions from the
* programming environment.
* results stored in NAOS_ENV, NMOS_ENV
*
* Obtain environments CORE energy, ECORE_ENV
*
* Jeppe Olsen, December 97
*              QDOTS inserted (again ), Feb 2003
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'lucinp.inc'
*
      IF(ENVIRO(1:6).EQ.'MOLCAS') THEN
        write(6,*) 'MOLCAS env!'
        CALL GETOBS2(ECORE_ENV,NAOS_ENV,NMOS_ENV)
      ELSE IF(ENVIRO(1:6).EQ.'DALTON' ) THEN
        write(6,*) 'DALTON env!'
        CALL GETOBS_DALTON(ECORE_ENV,NAOS_ENV,NMOS_ENV)
      ELSE IF(ENVIRO(1:5).EQ.'LUCIA') THEN
        write(6,*) 'LUCIA env!'
*. Lucia: core energy is obtained from 2-e file
        CALL GETOBS_LUCIA(NAOS_ENV,NMOS_ENV)
      ELSE IF(ENVIRO(1:4).EQ.'QDOT' ) THEN
        write(6,*) 'QDOT env!'
          CALL GETOBS_LUCIA(NAOS_ENV,NMOS_ENV)
          ECORE_ENV = 0.0D0
      ELSE IF(ENVIRO(1:4).EQ.'NONE') THEN
*. No environment, 
        WRITE(6,*) 'GET_ORB_DIM_ENV  in problems '
        WRITE(6,*) 'No ENVIRO parameter defined '
      END IF
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' From GET_ORB_FROM_ENV: '
        WRITE(6,*) ' ======================='
        WRITE(6,*) ' NAOS_ENV'
        CALL IWRTMA(NAOS_ENV,1,NSMOB,1,NSMOB)
        WRITE(6,*) ' NMOS_ENV'
        CALL IWRTMA(NMOS_ENV,1,NSMOB,1,NSMOB)
        WRITE(6,*) ' ECORE_ENV=', ECORE_ENV
      END IF
*
      RETURN
      END
      SUBROUTINE ZERORC(MBLOCK,IFIL,IAMPACKED)
*
* A record was known to be identical  zero
*
* Write corresponding info to file IFIL
*
* IAMPACKED added Oct. 98 / Jeppe Olsen
      IMPLICIT REAL*8 (A-H,O-Z)
      INTEGER ISCR(2)
* Zero record
      ISCR(1) = 1
*. Packed form
      ISCR(2) = IAMPACKED
*
      CALL ITODS(ISCR,2,2,IFIL)
*
      RETURN
      END
      SUBROUTINE MULT_BLOC_MAT(C,A,B,NBLOCK,LCROW,LCCOL,
     &                         LAROW,LACOL,LBROW,LBCOL,ITRNSP)
*
* Multiply two blocked matrices 
*
* ITRNSP = 0 => C = A * B
* ITRNSP = 1 => C = A(T) * B
* ITRNSP = 2 => C = A * B(T)
*
* Jeppe Olsen
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION A(*),B(*)
      INTEGER LCROW(NBLOCK),LCCOL(NBLOCK)
      INTEGER LAROW(NBLOCK),LACOL(NBLOCK)
      INTEGER LBROW(NBLOCK),LBCOL(NBLOCK)
*. Output
      DIMENSION C(*)
*
*. To get rid of annoying and incorrect compiler warnings
      IOFFA = 0
      IOFFB = 0
      IOFFC = 0
*
      DO IBLOCK = 1, NBLOCK
       IF(IBLOCK.EQ.1)  THEN
         IOFFA = 1
         IOFFB = 1
         IOFFC = 1
       ELSE
         IOFFA = IOFFA + LAROW(IBLOCK-1)*LACOL(IBLOCK-1)
         IOFFB = IOFFB + LBROW(IBLOCK-1)*LBCOL(IBLOCK-1)
         IOFFC = IOFFC + LCROW(IBLOCK-1)*LCCOL(IBLOCK-1)
       END IF
*
       ZERO = 0.0D0
       ONE = 1.0D0
       CALL MATML7(C(IOFFC),A(IOFFA),B(IOFFB),
     &             LCROW(IBLOCK),LCCOL(IBLOCK),
     &             LAROW(IBLOCK),LACOL(IBLOCK),
     &             LBROW(IBLOCK),LBCOL(IBLOCK),
     &             ZERO,ONE,ITRNSP)
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*)
        WRITE(6,*) ' output matrix from MULT_BLOC_MAT '
        WRITE(6,*) ' ================================='
        WRITE(6,*)
        CALL APRBLM2(C,LCROW,LCCOL,NBLOCK,0)
      END IF
*
      RETURN
      END 
      SUBROUTINE TRAN_SYM_BLOC_MAT(AIN,X,NBLOCK,LBLOCK,AOUT,SCR)
*
* Transform a blocked symmetric matrix AIN with blocked matrix
*  X to yield blocked matrix AOUT
*
* Aout = X(transposed) A X
*
* Jeppe Olsen
*
      IMPLICIT REAL*8(A,H,O-Z)
*. Input
      DIMENSION AIN(*),X(*),LBLOCK(NBLOCK)
*. Output 
      DIMENSION AOUT(*)
*. Scratch: At least twice the length of largest block 
      DIMENSION SCR(*)
*
*. To get rid of annoying and incorrect compiler warnings
      IOFFP = 0  
      IOFFC = 0
      DO IBLOCK = 1, NBLOCK
       IF(IBLOCK.EQ.1) THEN
         IOFFP = 1
         IOFFC = 1
       ELSE
         IOFFP = IOFFP + LBLOCK(IBLOCK-1)*(LBLOCK(IBLOCK-1)+1)/2
         IOFFC = IOFFC + LBLOCK(IBLOCK-1)** 2                     
       END IF
       L = LBLOCK(IBLOCK)
       K1 = 1
       K2 = 1 + L **2
*. Unpack block of A
C      TRIPAK(AUTPAK,APAK,IWAY,MATDIM,NDIM,SIGN)
       SIGN = 1.0D0
       CALL TRIPAK(SCR(K1),AIN(IOFFP),2,L,L)
*. X(T)(IBLOCK)A(IBLOCK)
       ZERO = 0.0D0
       ONE  = 1.0D0
       CALL MATML7(SCR(K2),X(IOFFC),SCR(K1),L,L,L,L,L,L,
     &             ZERO,ONE,1)
*. X(T) (IBLOCK) A(IBLOCK) X (IBLOCK)
       CALL MATML7(SCR(K1),SCR(K2),X(IOFFC),L,L,L,L,L,L,
     &             ZERO,ONE,0)
*. Pack and transfer
       CALL TRIPAK(SCR(K1),AOUT(IOFFP),1,L,L)
      END DO
*
      NTEST = 000
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' output matrix TRAN_SYM_BLOC_MAT '
        WRITE(6,*) ' ==============================='
        CALL APRBLM2(AOUT,LBLOCK,LBLOCK,NBLOCK,1)
      END IF
*
      RETURN
      END
      SUBROUTINE DIAG_GASBLKS(A,X,IGASL_INI,IGASL_FIN,NOBPTS_L,
     &                        SCR1,SCR2,IFORM)
*
* A packed matrix A over orbitals in symmetry-order
* and a orbital partitioning LGAS is given.
* Diagonalize diagonal blocks of A
*
*. IMAT_FORM = 1: A contains only active orbitals
*. IMAT_FORM = 2: A contains all orbitals
*. IMAT_FORM = 3: Use IGASL_INI,IGASL_FIN,NOBPTS_L to define partitioning
* 
*
* Input 
* =====
* A: Input matrix
* IGASL_INI, IGASL_FIN, NOBPTS_L: Info for local definition of partitioning
*                                 (Dummy if IFORM.ne.3)
*
* Output
* ======
* X: Eigenvector expansion, sorted according to eigenvalues in
*     each subspace
* A: Corresponding eigenvalues
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
*
      DIMENSION A(*),X(*)
      DIMENSION NOBPTS_L(IGASL_INI:IGASL_FIN,MXPOBS)
*. Scratch: Number of orbitals **2 ( atmost ) 
      DIMENSION SCR1(*), SCR2(*)
*. And an array giving number of orbitals per symmetry
      DIMENSION NOBPSM_G(MXPOBS)
*
      NTEST = 100
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' DIAG_BLKS'
        WRITE(6,*) ' ========='
        WRITE(6,*) ' NSMOB NGAS ', NSMOB,NGAS
      END IF
*
      IB_EIGENVAL = 1
      DO ISM = 1, NSMOB
        IF(IFORM.EQ.1) THEN
         NOBPSM_G(ISM) = NACOBS(ISM)
        ELSE IF(IFORM.EQ.2) THEN
         NOBPSM_G(ISM) = NTOOBS(ISM)
        ELSE
         NOBPSM_G(ISM) = 
     &   IELSUM(NOBPTS_L(IGASL_INI,ISM),IGASL_FIN-IGASL_INI+1)
        END IF
      END DO
      IF(NTEST.GE.1000) THEN
       WRITE(6,*) ' Number of orbitals per symmetry'
       CALL IWRTMA(NOBPSM_G,1,NSMOB,1,NSMOB)
      END IF
*
*. To get rid of annoying and incorrect compiler warnings
      IOFFMTP = 0
      IOFFMTC = 0
      IOFFOB = 0
*
      DO ISMOB = 1, NSMOB
*. Number of orbials of symmetry ISMOB
*. offset for symmetryblocks in matrices
        IF(ISMOB.EQ.1) THEN
          IOFFMTP = 1
          IOFFMTC = 1
        ELSE
          IOFFMTP = IOFFMTP + NOBPSM_G(ISMOB-1)*(NOBPSM_G(ISMOB-1)+1)/2
          IOFFMTC = IOFFMTC + NOBPSM_G(ISMOB-1) ** 2                    
        END IF
*. Zero symmetry block of eigenvector matrix to avoid interactions
*. between different blocks
        LOBPS = NOBPSM_G(ISMOB)
        ZERO = 0.0D0
        CALL SETVEC(X(IOFFMTC),ZERO,LOBPS**2)
*. Loop over subblocks, extract,  diagonalize, and expand 
        ITP_INI = -2810
        ITP_END = -2810
        IF(IFORM.EQ.1) THEN
          ITP_INI = 1
          ITP_END = NGAS
        ELSE IF(IFORM.EQ.2) THEN
          ITP_INI = 0
          ITP_END = NGAS + 1
        ELSE IF(IFORM.EQ.3) THEN
          ITP_INI = IGASL_INI
          ITP_END = IGASL_FIN
        END IF
        DO ITPOB = ITP_INI, ITP_END
          IF(ITPOB.EQ.ITP_INI) THEN
            IOFFOB=1
          ELSE
            IF(IFORM.LE.2) THEN
              IOFFOB = IOFFOB + NOBPTS_GN(ITPOB-1,ISMOB)
            ELSE
              IOFFOB = IOFFOB + NOBPTS_L(ITPOB-1,ISMOB)
            END IF
          END IF
          IF(IFORM.LE.2) THEN
            LOB = NOBPTS_GN(ITPOB,ISMOB)
          ELSE
            LOB = NOBPTS_L(ITPOB,ISMOB)
          END IF
*. Extract
          IJ2 = 0
          DO IOB = IOFFOB,IOFFOB+LOB-1
            JOBMX = IOB
            DO JOB = IOFFOB,JOBMX
              IJ1 = IOFFMTP -1 + IOB*(IOB-1)/2+JOB
              IJ2 = IJ2 + 1
              SCR1(IJ2) = A(IJ1)
            END DO
          END DO
          IF(NTEST.GE.1000) THEN
            WRITE(6,*) ' Extracted block of matrix for  ISMOB,ITPOB = ',
     &      ISMOB,ITPOB
            CALL PRSYM(SCR1,LOB)
          END IF
*, Diagonalize
C         CALL EIGEN(WORK(KMAT1-1+IOFFP),WORK(KMAT2-1+IOFFC),LORB,0,1)
          CALL EIGEN(SCR1, SCR2, LOB,0,1)
          CALL COPDIA(SCR1,A(IB_EIGENVAL),LOB,1)
          IB_EIGENVAL = IB_EIGENVAL + LOB

          IF(NTEST.GE.100) THEN
            WRITE(6,*) ' Eigenvalues'
            WRITE(6,*) (SCR1(I*(I+1)/2),I=1, LOB)
          END IF
          IF(NTEST.GE.1000) THEN
            WRITE(6,*) ' Eigenvalues and eigenvectors'
            WRITE(6,*)
            CALL WRTMAT(SCR2,LOB,LOB,LOB,LOB)
          END IF
*. Expand eigenvector to full symmetry block
          IJ2 = 0
          DO JOB = IOFFOB,IOFFOB+LOB-1
            DO IOB = IOFFOB,IOFFOB+LOB-1
              IJ1 = IOFFMTC -1 + (JOB-1)*LOBPS + IOB
              IJ2 = IJ2 + 1
              X(IJ1) = SCR2(IJ2)
            END DO
          END DO
        END DO
      END DO
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Complete eigenvector matrix'
        CALL APRBLM2(X,NOBPSM_G,NOBPSM_G,NSMOB,0)
      END IF
*
      RETURN
      END 
      SUBROUTINE ZERO_OFFBLKS(XIN,XOUT,LGAS,MXPOBS,NSMOB,NGAS,ISYM)
*
* A matrix XIN is given in symmetry blocked form, 
*. (total symmetrix )
*
*. XIN is assumed to contain only active orbitals
* 
* Copy from XIN to XOUT, the elements that belongs to
* identical orbital subspaces. Remaining elements of XOUT
* are set to zero
*
* The partitioning of orbitals is given by LGAS
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      DIMENSION XIN(*),XOUT(*)
      DIMENSION LGAS(MXPOBS,*)
*. Total number of elements
      NELMNT = 0
      DO ISMOB = 1, NSMOB
        NIORB = 0
        DO ITPOB = 1, NGAS
          NIORB = NIORB + LGAS(ISMOB,ITPOB)
        END DO
        IF(ISYM.EQ.0) THEN
          NELMNT = NELMNT + NIORB **2
        ELSE
          NELMNT = NELMNT + NIORB*(NIORB+1)/2
        END IF
      END DO
      WRITE(6,*) ' ZERO_OFFBLKS: Number of elements ', NELMNT
      ZERO = 0.0D0
      CALL SETVEC(XOUT,ZERO,NELMNT)
*. To get rid of annoying and incorrect compiler warnings
      IOFFMT = 0
      IOFFOB = 0
*
      DO ISMOB = 1, NSMOB
*. offset for symmetryblock in matrix 
        IF(ISMOB.EQ.1) THEN
          IOFFMT = 1
        ELSE
          LORB = 0
          DO ITPOB = 1, NGAS 
            LORB = LORB + LGAS(ISMOB-1,ITPOB)
          END DO
          IF(ISYM.EQ.0) THEN
            IOFFMT = IOFFMT + LORB ** 2
          ELSE
            IOFFMT = IOFFMT + LORB*(LORB+1)/2
          END IF
        END IF
*. Number of orbitals of this symmetry
        LOBSM = 0
        DO ITPOB = 1, NGAS
          LOBSM = LOBSM + LGAS(ISMOB,ITPOB)
        END DO
*
        DO ITPOB = 1, NGAS
          IF(ITPOB.EQ.1) THEN
            IOFFOB=1
          ELSE
            IOFFOB = IOFFOB + LGAS(ISMOB,ITPOB-1)
          END IF
          LOB = LGAS(ISMOB,ITPOB)
          DO IOB = IOFFOB,IOFFOB+LOB-1
            IF(ISYM.EQ.0) THEN
              JOBMX = IOFFOB+LOB-1
            ELSE
               JOBMX = IOB
            END IF
            DO JOB = IOFFOB,JOBMX
              IF(ISYM.EQ.0) THEN
                IJ = IOFFMT -1 + (JOB-1)*LOBSM+IOB
              ELSE
                IJ = IOFFMT -1 + IOB*(IOB-1)/2+JOB
              END IF
              WRITE(6,*) 'ISMOB ITPOB ', ISMOB,ITPOB
              WRITE(6,*) ' IOB JOB IJ ',IOB,JOB,IJ
              XOUT(IJ) = XIN(IJ)
            END DO
          END DO
        END DO
      END DO
*
COLD  NTEST = 100
COLD  IF(NTEST.GE.100) THEN
COLD    WRITE(6,*) ' Output matrix from ZERO_OFFBLOCKS...'
COLD    CALL APRBLM2(XOUT,NACOBS,NACOBS,NSMOB,ISYM)
COLD  END IF
*
      RETURN
      END 
      SUBROUTINE ZERO_OFFBLKS_OLD(XIN,XOUT,ISYM)
*
* A matrix XIN is given in symmetry blocked form, 
*. (total symmetrix )
*
*. XIN is assumed to contain only active orbitals
* 
* Copy from XIN to XOUT, the elements that belongs to
* identical orbital subspaces. Remaining elements of XOUT
* are set to zero
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
*
      DIMENSION XIN(*),XOUT(*)
*
*. To get rid of annoying and incorrect compiler warnings
      IOBOFF = 0
      IMTOFF = 0
*
      DO ISMOB = 1, NSMOB
        IF(ISMOB.EQ.1) THEN
          IOBOFF = 1
          IMTOFF = 1
        ELSE 
          IOBOFF = IOBOFF + NACOBS(ISMOB-1)
          IF(ISYM.GE.1) THEN
            IMTOFF = IMTOFF + NACOBS(ISMOB-1)*(NACOBS(ISMOB-1)+1)/2
          ELSE 
            IMTOFF = IMTOFF + NACOBS(ISMOB-1)**2
          END IF
        END IF
        LORB = NACOBS(ISMOB)
        IJ = 0
        DO IORB = IOBOFF,IOBOFF+LORB-1
          IF(ISYM.LE.0) THEN
            JORBMX = IOBOFF+LORB-1
          ELSE
            JORBMX = IORB
          END IF
          DO JORB = IOBOFF,JORBMX
            IJ = IJ + 1
            WRITE(6,*) 'ISYM IORB JORB ',ISYM,IORB,JORB
            WRITE(6,*) ' IOBTP JOBTP ',ITPFSO(IORB),ITPFSO(JORB)
            IF(ITPFSO(IORB).EQ.ITPFSO(JORB)) THEN
              XOUT(IMTOFF-1+IJ) = XIN(IMTOFF-1+IJ)
            ELSE
              XOUT(IMTOFF-1+IJ) = 0.0D0
            END IF
          END DO
        END DO
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Output matrix from ZERO_OFFBLOCKS...'
        CALL APRBLM2(XOUT,NACOBS,NACOBS,NSMOB,ISYM)
      END IF
*
      RETURN
      END 
      SUBROUTINE TYPE_TO_SYM_REO_MAT(XIN,XOUT)
*
*. a matrix XIN is given as NTOOB X NTOOB matrix in type form
*
* If ISYM.eq.1 matrix is assuemed to be packed - lower half as usual
*. 
*. Reorder to symmetry-ordered and -blocked matrix to give XOUT
*
*. Matrix is assumed to exclude inactive orbitals !!
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
*.
      DIMENSION XIN(NTOOB,NTOOB)
      DIMENSION XOUT(*)
*
      NTEST = 0
*
*. To get rid of annoying and incorrect compiler warnings
      IOBOFF = 0
      IMTOFF = 0
*
      DO ISMOB = 1, NSMOB
*. IOBOFF: Offset for active orbitals in symmetry order
        IF(ISMOB.EQ.1) THEN
          IOBOFF = NINOBS(1)+1
          IMTOFF = 1
        ELSE
          IOBOFF =
     &    IOBOFF + NTOOBS(ISMOB-1)-NINOBS(ISMOB-1)+NINOBS(ISMOB)
          IMTOFF = IMTOFF + NACOBS(ISMOB-1)**2
        END IF
        LOB = NACOBS(ISMOB)
*
*. Extract symmetry block of matrix
*
        DO IOB = IOBOFF,IOBOFF + LOB-1
           DO JOB = IOBOFF, IOBOFF + LOB-1         
*. Corresponding type indeces
             IOBP = IREOST(IOB)
             JOBP = IREOST(JOB)
             XOUT(IMTOFF-1+(JOB-IOBOFF)*LOB+IOB-IOBOFF+1)
     &     = XIN(IOBP,JOBP)
           END DO
        END DO
*
      END DO
*. (End of loop over orbital symmetries )
      IF(NTEST.GE.10 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' Symmetry ordered matrix '                 
        WRITE(6,*) ' ======================='
        WRITE(6,*)
        CALL APRBLM2(XOUT,NACOBS,NACOBS,NSMOB,0)      
      END IF
*
      RETURN
      END
      SUBROUTINE MOROT(IMO)
*
* A MO-MO rotation matrix is given in KMOMO. Obtain
* final MO-MO rotation matrix by defining internal rotations
*
*
* Type of active orbitals is provided by the keyword IMO
*
* IMO = 1 => Natural orbitals
* IMO = 2 => Canonical orbitals
* IMO = 3 => Pseudo-natural orbitals
* IMO = 4 => Pseudo-canonical orbitals
* IMO = 5 => Psedo-natural-cannonical orbitals
*
* The inactive and secondary orbitals are in general defined
* as canonical orbitals
*
* Jeppe Olsen, Feb. 2011: From rewritten with inactive + sec orbitals added
*
* Expansion of current MO's in initial MO's is assumed in KMOMO
* Final MO-AO expansion stored in KMOAO
*       MO-MO expansion stored in KMOMO
*
* If no mo-ao trans is present, only, MOMO matrix is returned

      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'cintfo.inc'
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*)
        WRITE(6,*) ' ================'
        WRITE(6,*) ' MOROT in action '
        WRITE(6,*) ' ================'
        WRITE(6,*)
      END IF
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' IMO parameter ', IMO
        WRITE(6,*) ' INTIMP = ', INTIMP
      END IF
      IF(NTEST.GE.1) THEN
        IF(IMO.EQ.1) THEN
          WRITE(6,*) ' Final orbitals: natural orbitals '
        ELSE IF (IMO.EQ.2) THEN
          WRITE(6,*) ' Final orbitals: canonical orbitals '
        ELSE IF (IMO.EQ.3) THEN
          WRITE(6,*) ' Final orbitals: pseudo-natural orbitals '
        ELSE IF (IMO.EQ.4) THEN
          WRITE(6,*) ' Final orbitals: pseudo-canonical orbitals '
        ELSE IF (IMO.EQ.4) THEN
          WRITE(6,*) 
     &    ' Final orbitals: pseudo-natural-canonical orbitals'
        END IF
      END IF
*
      IF(NTEST.GE.100) THEN
       WRITE(6,*) ' Initial MOMO matrix '
       CALL APRBLM2(WORK(KMOMO),NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'MOROT ')
      CALL MEMMAN(KLMAT1,NTOOB**2,'ADDL  ',2,'MAT1  ')
      CALL MEMMAN(KLMAT2,NTOOB**2,'ADDL  ',2,'MAT2  ')
      CALL MEMMAN(KLMAT2C,NTOOB**2,'ADDL  ',2,'MAT2C ')
      CALL MEMMAN(KLMAT3,NTOOB**2,'ADDL  ',2,'MAT3  ')
      CALL MEMMAN(KLMAT4,2*NTOOB**2,'ADDL  ',2,'MAT4  ')
      CALL MEMMAN(KLMAT5,NTOOB**2,'ADDL  ',2,'MAT5  ')
*
      LMOMO = NDIM_1EL_MAT(1,NTOOBS,NTOOBS,NSMOB,0)
      LACAC = NDIM_1EL_MAT(1,NACOBS,NACOBS,NSMOB,0)
C?    WRITE(6,*) ' LMOMO, LACAC = ', LMOMO, LACAC
C             NDIM_1EL_MAT(IHSM,NRPSM,NCPSM,NSM,IPACK)
*. Start be setting up the FI + FA matrix defining the inactive and
*. secondary orbitals
*. Construct FI+FA in WORK(KLMAT1), using integrals in initial basis
      KINT2 = KINT_2EINI
      IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_FULL))
      KINT2_FSAVE = KINT2_A(IE2ARR_F)
      KINT2_A(IE2ARR_F) = KINT_2EINI
      KINT2_A(IE2ARR_F) = KINT_2EMO
*
      CALL FI_FROM_INIINT(WORK(KFI),WORK(KMOMO),WORK(KH),
     &                      ECORE_HEX,1)
      IF(NTEST.GE.100) THEN
      WRITE(6,*) ' Inactive Fock matrix '
      CALL APRBLM2(WORK(KFI),NTOOBS,NTOOBS,NMSOB,1)
      END IF
      CALL FA_FROM_INIINT
     &(WORK(KFA),WORK(KMOMO),WORK(KMOMO),WORK(KRHO1),1)
      KINT2_A(IE2ARR_F)  = KINT2_FSAVE
*
      ONE = 1.0D0
      CALL VECSUM(WORK(KLMAT1),WORK(KFI),WORK(KFA),ONE,ONE,NINT1)
      IF(IMO.EQ.5) CALL COPVEC(WORK(KLMAT1),WORK(KLMAT5),NINT1)
*
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' FIFA matrix '
        CALL APRBLM2(WORK(KLMAT1),NTOOBS,NTOOBS,NSMOB,1)
      END IF
      IF(IMO.EQ.2) THEN
* Diagonalize symmetry blocks of FIFA to obtain canonical orbitals
        CALL DIAG_GASBLKS(WORK(KLMAT1),WORK(KLMAT2C),1,1,NTOOBS,
     &                       WORK(KLMAT3),WORK(KLMAT4),3)
        IF(NTEST.GE.10) THEN
          WRITE(6,*) ' Canonical orbital energies'
          CALL PRINT_SCALAR_PER_ORB(WORK(KLMAT1),NTOOBS)
        END IF
      ELSE
*. Diagonize symmetry-type blocks of FIFA to obtain pseudo canonical
*. orbitals
      CALL DIAG_GASBLKS(WORK(KLMAT1),WORK(KLMAT2C),IDUM,IDUM,IDUM,
     &                     WORK(KLMAT3),WORK(KLMAT4),2)
      END IF
      IF(NTEST.GE.100) THEN
       WRITE(6,*)  ' Expansion of final orbitals, canonical part'
       CALL APRBLM2(WORK(KLMAT2C),NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      IF(IMO.EQ.1.OR.IMO.EQ.3.OR.IMO.EQ.5) THEN
*. Symmetry ordered density matrix  over active orbitals
C       REORHO1(RHO1I,RHO1O,IRHO1SM,IWAY)
        CALL REORHO1(WORK(KRHO1),WORK(KLMAT2),1,1)
COLD    CALL EXTR_SYMBLK_ACTMAT(WORK(KRHO1),WORK(KLMAT2),1)
COLD    CALL TYPE_TO_SYM_REO_MAT(WORK(KRHO1),WORK(KLMAT2))
*. Pack to triangular form
C       TRIPAK_BLKM(AUTPAK,APAK,IWAY,LBLOCK,NBLOCK)
        CALL TRIPAK_BLKM(WORK(KLMAT2),WORK(KLMAT1),1,
     &       NACOBS,NSMOB)
        LEN_AC = NDIM_1EL_MAT(1,NACOBS,NACOBS,NSMOB,1)
        ONEM = -1.0D0
        CALL SCALVE(WORK(KLMAT1),ONEM,LEN_AC)
        IF(NTEST.GE.100) THEN
          WRITE(6,*) ' Packed density matrix ( times - 1 )'
          CALL APRBLM2(WORK(KLMAT1),NACOBS,NACOBS,NSMOB,1)
        END IF
*
        IF (IMO.EQ.1.OR.IMO.EQ.5) THEN
*. Diagonalize symmetryblocks of density matrix over active orbitals  
*  - mixes different  spaces
         CALL DIAG_GASBLKS(WORK(KLMAT1),WORK(KLMAT2),1,1,NACOBS,
     &                  WORK(KLMAT3),WORK(KLMAT4),3)
        ELSE IF (IMO.EQ.3) THEN
*. Diagonalize type-symmetry blocks of density matrix over active blocks
         CALL DIAG_GASBLKS(WORK(KLMAT1),WORK(KLMAT2),IDUM,IDUM,IDUM,
     &                  WORK(KLMAT3),WORK(KLMAT4),1)
C             DIAG_GASBLKS(A,X,IGASL_INI,IGASL_FIN,NOBPTS_L,SCR1,SCR2,IFORM)
*. WORK(KLMAT2) contains eigenvector expansions
        END IF
*
        IF(NTEST.GE.100) THEN
         WRITE(6,*)  ' Expansion of (pseudo-natural) orbitals'
         CALL APRBLM2(WORK(KLMAT2),NACOBS,NACOBS,NSMOB,0)
        END IF
*
*. pseudo-natural-canonical orbitals
*
        IF( IMO .EQ. 5 ) THEN
*. Transform the active part of FIFA to pseudo natural basis
*. Extract active blocks of FIFA and save in KLMAT4
C              EXT_CP_AC_GASBLKS(NSMOB,NGAS,NOBPTS_GN,MXPNGAS,
C    &           IEORC,NTOOBS, NACOBS,AACT,AALL)
          CALL EXT_CP_AC_GASBLKS(NSMOB,NGAS,NOBPTS_GN,MXPNGAS,
     &         1,NTOOBS,NACOBS,WORK(KLMAT2),WORK(KLMAT5))
C         TRAN_SYM_BLOC_MAT(AIN,X,NBLOCK,LBLOCK,AOUT,SCR)
          CALL TRAN_SYM_BLOC_MAT(WORK(KLMAT4),WORK(KLMAT2),NSMOB,NACOBS,
     &                         WORK(KLMAT3),WORK(KLMAT5))
*. Transformed FIFA is now in WORK(KLMAT3)
*. Diagonalize
C              DIAG_GASBLKS(A,X,IGASL_INI,IGASL_FIN,NOBPTS_L,SCR1,SCR2,IFORM)
          CALL DIAG_GASBLKS(WORK(KLMAT3),WORK(KLMAT5),1,NPSSPC,NPSSH,
     &                 WORK(KLMAT1),WORK(KLMAT2),3)
*. Second transformation matrix is now in WORK(KLMAT5)
*. Obtain total transformation matrix
C         MULT_BLOC_MAT(C,A,B,NBLOCK,LCROW,LCCOL,LAROW,LACOL,LBROW,LBCOL,ITRNSP)
          CALL MULT_BLOC_MAT(WORK(KLMAT4),WORK(KLMAT2),WORK(KLMAT5),NSMOB,
     &                     NACOBS,NACOBS,NACOBS,NACOBS,NACOBS,NACOBS,0)
*. We now have the transformation for the active orbitals in KLMAT4
          CALL COPVEC(WORK(KLMAT4),WORK(KLMAT2),LACAC)
        END IF ! (IMO=5)
*       ^ End if PS_NatCan orbital
*. Add the act-act transformation to the total MOMO transformation matrix
C           EXT_CP_AC_GASBLKS(NSMOB,NGAS,NOBPTS_GN,MXPNGAS,
C    &           IEORC,NTOOBS, NACOBS,AACT,AALL)
       CALL EXT_CP_AC_GASBLKS(NSMOB,NGAS,NOBPTS_GN,MXPNGAS,
     &      2,NTOOBS,NACOBS,WORK(KLMAT2),WORK(KLMAT2C))
       IF(NTEST.GE.100) THEN
         WRITE(6,*) ' MOMO transformation with natural part copied in'
         CALL APRBLM2(WORK(KLMAT2C),NTOOBS,NTOOBS,NSMOB,0)
       END IF
      END IF
*     ^ end if some kind of natural orbitals were involved
*
* Obtain total MOMO transformation and save in MOMO
*
*. And new MO-coefficients
      CALL MULT_BLOC_MAT(WORK(KLMAT2),WORK(KMOMO),WORK(KLMAT2C),
     &     NSMOB,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,0)
      CALL COPVEC(WORK(KLMAT2),WORK(KMOMO),LMOMO)
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MO-MO transformation matrix MOMO '
        CALL APRBLM2(WORK(KMOMO),NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      IF(NOMOFL.EQ.0) THEN
*
*. Obtain input MO-AO transformation matrix in KMOAOUT
C?    CALL GET_CMOAO_ENV(WORK(KMOAOIN))
      CALL MULT_BLOC_MAT(WORK(KMOAOUT),WORK(KMOAOIN),WORK(KMOMO),
     &       NSMOB,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,0)
*
      CALL COPVEC(WORK(KMOAOUT),WORK(KMOAO_ACT),LMOMO)
      IF(NTEST.GE.1) THEN
         WRITE(6,*) ' Output set of MO''s in required form'
         CALL PRINT_CMOAO(WORK(KMOAOUT))
      END IF
*. Save on file LUMOUT
      CALL PUTMOAO(WORK(KMOAOUT))
      END IF
*     ^ End if MOAO file is present 
*
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'MOROT ')
      RETURN
      END
      SUBROUTINE PUTMOAO(CMOAO) 
*
* SAVE   MOAO matrix CMOAO on LUMOUT
*
* A sunny day in April 96
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'crun.inc'
*
      IF(ENVIRO(1:6).EQ.'MOLCAS') THEN
*. MOLCAS environment               
        WRITE(6,*) ' PUTMOAO: MOLCAS environment'
        CALL PUTMOAO_MOLCAS(CMOAO,LUMOUT)
      ELSE
        WRITE(6,*) ' MOs will not be written to file in PUTMOAO'
      END IF
*
      RETURN
      END 
      SUBROUTINE PUTMOAO_MOLCAS(CMOAO,LU)
*
* WRITE MOAO matrix CMOAO on file LU in MOLCAS LUMORB format

*
* GETOBS assumed called to define /MOLOBS/
*
      IMPLICIT REAL*8(A-H,O-Z)
CTOBE?CHARACTER*80 TITLEMO
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
*     
      LOCC = 0
*. Full NBAS X NBAS assumed 
      CALL WRVEC(LU,NSYM,NBAS,NBAS,CMOAO,OCC,LOCC,
     &           ' MO orbitals obtained from LUCIA ')
      WRITE(6,*) ' Mo coefficients written to ', LU   
*
      RETURN
      END 
      SUBROUTINE GETMOAO(CMOAO)
*
* Obtain MOAO matrix and save in CMOAO
*
* A sunny day in April 96
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'crun.inc'
*
      IF(INTIMP.EQ.1) THEN
*. MOLCAS environment               
        WRITE(6,*) ' GETMOAO: MOLCAS environment'
        CALL GETMOAO_MOLCAS(CMOAO,LUMOIN)
      END IF
*
      RETURN
      END 
      SUBROUTINE GETMOAO_MOLCAS(CMOAO,LU)
*
* THE MO-AO file is assumed to be a NBAS X NBAS file in LUMORB format
* as delivered by SCF or RASREAD 

*
* Obtain MOAO transformation matrix from 
* MOLCAS file 
*
* GETOBS assumed called to define /MOLOBS/
*
      IMPLICIT REAL*8(A-H,O-Z)
      CHARACTER*80 TITLEMO
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
      
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' Information available in GETMOAO_MOLCAS'
        WRITE(6,*) ' ======================================='
        WRITE(6,*)
        WRITE(6,*) ' NSYM = ', NSYM
        WRITE(6,*) ' NBAS: '
        CALL IWRTMA(NBAS,1,NSYM,1,NSYM)
        WRITE(6,*) 'NORB: ' 
        CALL IWRTMA(NORB,1,NSYM,1,NSYM)
        WRITE(6,*) 'NFRO: ' 
        CALL IWRTMA(NFRO,1,NSYM,1,NSYM)
      END IF
*
      LOCC = 0
*. Full NBAS X NBAS matrix assumed, truncation only in in int transformation
      CALL RDVEC(LU,NSYM,NBAS,NBAS,CMOAO,OCC,LOCC,TITLEMO)
      WRITE(6,*) ' Header from MOAO file (LUMOIN)'
      WRITE(6,'(80A)') TITLEMO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Input MOAO transformation matrix '
        CALL APRBLM2(CMOAO,NBAS,NBAS,NSYM,0)
      END IF
*
      RETURN
      END 
      SUBROUTINE SCLH2(XLAMBDA)
*
*. Scale two electron integrals
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'cintfo.inc'
*
      CALL SCALVE(WORK(KINT2),XLAMBDA,NINT2)
*
      RETURN
      END 
      SUBROUTINE GENH1(XLAMBDA_X)
*
* Construct the general one-electron operator
*
* H = XLAMBDA*H(NORMAL) + (1-XLAMBDA)FIFA
*
*
* Where H(Normal) is the normal one-electron operator
* and FIFA is the sum of the inactive and active Fock matrices 
* used in CASPTN theory
*
* The correct one-electron density is assumed in place
*
* Jeppe Olsen, March 1996
*
*. Note: Correct Lambda is transferred through CRUN as of Feb. 98
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cecore.inc'
*
*. Construct FIFA in WORK(KFI)
*
      IF(IPART.NE.3) THEN
*. Normal M-P Partitioning
        WRITE(6,*) 'Normal MP partitioning'
        WRITE(6,*) 'Normal MP partitioning'
        WRITE(6,*) 'Normal MP partitioning'
        CALL COPVEC(WORK(KINT1O),WORK(KFI),NINT1)
        CALL FIFAM(WORK(KFI))
        CALL COPVEC(WORK(KFI),WORK(KFIO),NINT1)
        ECORE_H = 0.0D0
        IF(IUSE_PH.EQ.1) THEN
          CALL FI(WORK(KFI),ECORE_H,0)
          ECORE = ECORE_ORIG + ECORE_H 
        END IF
*. And write to disc
        LU18 = IGETUNIT(18)
        REWIND (LU18)
        WRITE(6,*) ' H0 written to disc '
        CALL TODSC(WORK(KFI),NINT1,-1,LU18)
      ELSE IF(IPART.EQ.3) THEN
        WRITE(6,*) ' Zero-order Hamiltonian readin '
        WRITE(6,*) ' Zero-order Hamiltonian readin '
        WRITE(6,*) ' Zero-order Hamiltonian readin '
        WRITE(6,*) ' Zero-order Hamiltonian readin '
        LU18 = IGETUNIT(18)
        REWIND (LU18)
        CALL FRMDSC(WORK(KFI),NINT1,-1,LU18,IMZERO,IAMPACK)
      END IF
*
*. And obtain modified operator in KINT1 ( No return !! )
*
      FAC2 = 1.0D0 - XLAMBDA
      CALL VECSUM(WORK(KINT1),WORK(KINT1),WORK(KFI),XLAMBDA,FAC2,
     &            NINT1)
*
      CALL COPVEC(WORK(KINT1),WORK(KINT1O),NINT1)
      NTEST = 00
      IF(NTEST.GE.100) THEN
       WRITE(6,*) 'Modified matrix as delivered by GENH1 '
       WRITE(6,*) '======================================'
       CALL APRBLM2(WORK(KINT1),NTOOBS,NTOOBS,NSMOB,1)
      END IF
*
      RETURN
      END 
      SUBROUTINE MIXHONE(H1,H2,NREPTP,IREPTP,NOBTP,NSMOB)
*
* Replace selected type blocks of H1 with the corresponding blocks 
* in H2
*
*. H1 and H2 are assumed to be in symmetry order !
*. -and total symmetric
*
*     Jeppe Olsen, March 14 1996 ( Still snowing in Lund )
*
      IMPLICIT REAL*8(A-H,O-Z)
*. General input
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*. Specific input
      DIMENSION IREPTP(*)
      DIMENSION H2(*)
*. Input and output 
      DIMENSION H1(*)
*. To eliminate annoying and incorrect compiler warnings 
      IOFF = 0
      IOBOFF = 0
      JOBOFF = 0
* 
      DO ISMOB = 1, NSMOB
        IF (ISMOB.EQ.1) THEN
          IOFF = 1
        ELSE
          IOFF = IOFF + NTOOBS(ISMOB-1)*(NTOOBS(ISMOB-1)+1)/2
        END IF
*. Loop over types for given symmetry
        DO ITP = 1, NOBTP
          IF(ITP.EQ.1) THEN
           IOBOFF = 1
          ELSE 
            IOBOFF = IOBOFF + NOBPTS(ITP-1,ISMOB)
          END IF
          DO JTP = 1, ITP
            IF(JTP.EQ.1) THEN
             JOBOFF = 1
            ELSE 
              JOBOFF = JOBOFF + NOBPTS(JTP-1,ISMOB)
            END IF
*. Number of elements in this type-type block
            LIOB = NOBPTS(ITP,ISMOB)
            LJOB = NOBPTS(JTP,ISMOB)
*
*. Should this block of H1 be replaced by corresponding block of H2
            IF(ITP.EQ.JTP) THEN
              IMOVE = 0
              DO KTP = 1, NREPTP
                IF(IREPTP(KTP).EQ.ITP) IMOVE = 1
              END DO
*
              IF(IMOVE.EQ.1) THEN
C?              WRITE(6,*) ' Block transfer ISMOB ITP JTP ',
C?   &          ISMOB,ITP,JTP
                DO IOB = IOBOFF,IOBOFF+LIOB-1
                  DO JOB = JOBOFF, IOB
                    H1(IOFF-1+IOB*(IOB-1)/2+JOB) 
     &            = H2(IOFF-1+IOB*(IOB-1)/2+JOB)
                  END DO
                END DO
              END IF
*
            END IF
          END DO
        END DO
      END DO
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*)
        WRITE(6,*) ' =================='
        WRITE(6,*) ' MIXHONE in action '
        WRITE(6,*) ' =================='
        WRITE(6,*)
        WRITE(6,*) ' NSMOB NOBTP ', NSMOB,NOBTP
        WRITE(6,*) ' Types to be changed '
        CALL IWRTMA(IREPTP,1,NREPTP,1,NREPTP)
        WRITE(6,*) ' output H1 and H2 '
C       APRBLM2(A,LROW,LCOL,NBLK,ISYM)
        CALL APRBLM2(H1,NTOOBS,NTOOBS,NSMOB,1)
        CALL APRBLM2(H2,NTOOBS,NTOOBS,NSMOB,1)
      END IF
*
      RETURN
      END 
      SUBROUTINE REPBLKS(VEC1,VEC2,LROW,LCOL,NBLK,ISYM,NREPBLK,IREPBLK)
*
* Two blocked vectors, VEC1 and VEC2 are given-
* Replace selected blocks in VEC1 with the same blocks in VEC2
*
* Jeppe Olsen
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION VEC1(*),VEC2(*)
      DIMENSION LROW(*),LCOL(*)
      DIMENSION IREPBLK(*)
*
      DO IBLK = 1, NREPBLK
*.Offset for IREPBLK(IBLK)
        IOFF = 1
        DO JBLK = 1,  IREPBLK(IBLK)-1
          IF(ISYM.GT.0) THEN
            IOFF =  IOFF + LROW(JBLK)*(LROW(JBLK)-1)/2
          ELSE
            IOFF = IOFF + LROW(JBLK)*LCOL(JBLK)
          END IF
        END DO
*. Number of elements in block
        IF(ISYM.GT.0) THEN
          LENGTH = LROW(IREPBLK(IBLK))*( LROW(IREPBLK(IBLK))+1)/2
        ELSE
          LENGTH = LROW(IREPBLK(IBLK)) * LCOL(IREPBLK(IBLK))
        END IF
*. and copy
        CALL COPVEC(VEC2(IOFF),VEC1(IOFF),LENGTH)
      END DO
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*)
        WRITE(6,*) '  Output from REPBLKS '
        WRITE(6,*) ' ====================='
        WRITE(6,*)
        WRITE(6,*) ' Blocks to be copied '
        CALL IWRTMA(IREPBLK,1,NREPBLK,1,NREPBLK)
        WRITE(6,*)
        WRITE(6,*) ' Updated matrix '
C       APRBLM2(A,LROW,LCOL,NBLK,ISYM)
        CALL APRBLM2(VEC2,LROW,LCOL,NBLK,ISYM)
      END IF
*
      RETURN
      END
      SUBROUTINE ZIRAT
*
* Ratio between real and integer
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER*8 :: ITEST
*. Input
      INCLUDE 'machine.inc'
*. Output
      INCLUDE 'irat.inc'
*. For CRAY or explicit 64 bit architecture: IRAT = 1
      IF(MACHINE(1:4).EQ.'CRAY'.OR.MACHINE(1:2).EQ.'64') THEN
        IRAT = 1
      ELSE
*. For all other architectures: 2 integers per real
        IRAT = 2
      END IF
*
* better make a test:
c not all compilers know this one
c      ITEST = #FFFFFFFFFFFFFFFF
c so take the old-fashined syntax
C     ITEST = Z'FFFFFFFFFFFFFFFF'
C     CALL ISETVC(ITEST,0,1)
C     IF (ITEST.EQ.0) THEN
C       IRAT = 1
c dto.
c      ELSE IF (ITEST.EQ.#00000000FFFFFFFF.OR.
c     &         ITEST.EQ.#FFFFFFFF00000000) THEN
C     ELSE IF (ITEST.EQ.Z'00000000FFFFFFFF'.OR.
C    &         ITEST.EQ.Z'FFFFFFFF00000000') THEN
C       ! outcome depends on high-word/low-word ordering, but this
C       ! is not important for us
C       IRAT = 2
C     ELSE
C       WRITE(6,*) 'Silly outcome in ZIRAT, d''you run on a C64?', ITEST
C       STOP 'ZIRAT'
C     END IF
*
      WRITE(6,*)
      WRITE(6,'(1H ,6X,A,I2)') 
     &'Ratio between Integer and Real word length ', IRAT
      WRITE(6,*)
*
      RETURN
      END
      SUBROUTINE ORBINH1(IORBINH1,IORBINH1_NOCCSYM,NTOOBS,NTOOB,NSMOB)
*
* Obtain array of 2 orbital indeces,
* for symmetry packed matrices
*
* IORBINH1: Lower half packed
* IORBINH1_NOCCSYM: Complete blocks
*
* resulting indeces are with respect to start of given symmetry block
* while input orbital indeces are absolute and in symmetry order
*
* Jeppe Olsen, March 1995
*              ORBINH1_NOCCSYM added August 2000
*              ITRA_ROUTE added, May 2011
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'crun.inc'
*. Input
      DIMENSION NTOOBS(NSMOB)
*. output
      DIMENSION IORBINH1(NTOOB,NTOOB), IORBINH1_NOCCSYM(NTOOB,NTOOB)
*
C?    WRITE(6,*) ' ORBINH1 speaking '
C?    WRITE(6,*) ' NSMOB NTOOB ',NSMOB,NTOOB
C?    WRITE(6,*) ' NTOOBS '
C?    CALL IWRTMA(NTOOBS,1,NSMOB,1,NSMOB)
*. To eliminate annoying and incorrect compiler warnings 
      IOFF = 0
      JOFF = 0
      INDEX = 0

*. Loop over symmetries of orbitals

      DO ISM = 1, NSMOB
        IF(ISM.EQ.1) THEN
          IOFF = 1
        ELSE
          IOFF = IOFF + NTOOBS(ISM-1)
        END IF
        DO JSM = 1, NSMOB
          IF(JSM.EQ.1) THEN
            JOFF = 1
          ELSE
            JOFF = JOFF + NTOOBS(JSM-1)
          END IF
C?        WRITE(6,*) ' ISM JSM IOFF JOFF', ISM,JSM,IOFF,JOFF
          DO IORB = 1, NTOOBS(ISM)
            IABS = IOFF -1 + IORB
            DO JORB = 1, NTOOBS(JSM)
              JABS = JOFF -1 + JORB
C?            write(6,*) ' IORB JORB IABS JABS ',IORB,JORB,IABS,JABS
              IF(ISM.GT.JSM) THEN
                IF(ITRA_ROUTE.EQ.1) THEN
                  INDEX = (IORB-1)*NTOOBS(JSM) + JORB
                ELSE 
                  INDEX = (JORB-1)*NTOOBS(ISM) + IORB
                END IF
              ELSE IF(ISM.EQ.JSM) THEN
                IF(IORB.GE.JORB) THEN
                  INDEX = IORB*(IORB-1)/2 + JORB
                ELSE
                  INDEX = JORB*(JORB-1)/2 + IORB
                END IF
              ELSE IF(ISM.LT.JSM) THEN
                IF(ITRA_ROUTE.EQ.1) THEN
                  INDEX = (JORB-1)*NTOOBS(ISM) + IORB
                ELSE
                  INDEX = (IORB-1)*NTOOBS(JSM) + JORB
                END IF
              END IF
              IF(ITRA_ROUTE.EQ.1) THEN
                INDEX_NOCCSYM = (IORB-1)*NTOOBS(JSM) + JORB
              ELSE
                INDEX_NOCCSYM = (JORB-1)*NTOOBS(ISM) + IORB
              END IF
              IORBINH1(IABS,JABS) = INDEX
              IORBINH1_NOCCSYM(IABS,JABS) = INDEX_NOCCSYM
            END DO
          END DO
*. End of loops over orbital indeces
        END DO
      END DO
*. End of loop over orbital symmetries
*
      NTEST = 00
      IF(NTEST .GE. 100 ) THEN
        WRITE(6,*) ' IORBINH1 matrix delivered from ORBINH1'
        CALL IWRTMA(IORBINH1,NTOOB,NTOOB,NTOOB,NTOOB)
      END IF
*
      RETURN
      END
      SUBROUTINE GASCI(ISM,ISPC,IPRNT,IIUSEH0P,MPORENP_E,
     &                 EREF,ERROR_NORM_FINAL,CONV_F)
*
* CI optimization in GAS space number ISPC for symmetry ISM              
*
*
* Jeppe Olsen, Winter of 1995
*
*                    Oct. 30, 2012; Jeppe Olsen; call to Z_BLKFO changed
*                    Nov. 9,  2012; Jeppe Olsen; call to CIEG5 changed 
*                                                (KSIBT,.. added for Pico)
*                    Jan. 6, 2013; Jeppe Olsen; Improved preconditioner (ISBSPPR added)
*                    Feb. 12, 2013: IROOT_SEL replacing ROOT_HOMING
*                    Feb. 25, 2013: Root selection Modified
* Last modification; July 13, 2013; Jeppe Olsen; Subspace H0 reinstated
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
      INCLUDE 'wrkspc.inc'
      LOGICAL CONV_F
      EXTERNAL MV7
      INCLUDE 'cicisp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'csm.inc' 
      INCLUDE 'cstate.inc' 
      INCLUDE 'crun.inc'
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'intform.inc'
      INCLUDE 'comjep.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'spinfo.inc'
*
*. Common block for communicating with sigma
      INCLUDE 'cands.inc'
*
      INCLUDE 'cecore.inc'
      COMMON/CMXCJ/MXCJ,MAXK1_MX,LSCMAX_MX
*
      COMMON/H_OCC_CONS/IH_OCC_CONS
*
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'GASCI ')
      NTEST = 0
      NTEST = 5
      NTEST = MAX(NTEST,IPRNT)
      MXACJ = 0
      MXACIJ = 0
      MXAADST = 0
*. Normal integrals accessed
      IH1FORM = 1
      I_RES_AB = 0
      IH2FORM = 1
*. CI not CC
      ICC_EXC = 0
*. Not just number conserving part
      IH_OCC_CONS_TEST = 0
      IF(IH_OCC_CONS_TEST.EQ.1) THEN
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        WRITE(6,*) ' IH_OCC_CONS set to one in GASCI '
        IH_OCC_CONS = 1
      END IF
*
      IF(NTEST.GT.1) THEN
        WRITE(6,*) 
        WRITE(6,*) ' ====================================='
        WRITE(6,*) ' Control has been transferred to GASCI'
        WRITE(6,*) ' ====================================='
        WRITE(6,*) 
        WRITE(6,*) ' NROOT, IROOT_SEL = ',NROOT,  IROOT_SEL
C?      WRITE(6,*) ' IIUSEH0P = ', IIUSEH0P
C?      WRITE(6,*) ' MPORENP_E = ', MPORENP_E
      END IF
      IF(NTEST.GE.5) THEN
        WRITE(6,'(A)') '  A few pertinent data: '
        WRITE(6,*)
        WRITE(6,'(A,I2)') '  CI space         ',ISPC
        WRITE(6,*)
        WRITE(6,*) ' Number of GAS spaces included ',LCMBSPC(ISPC)
        WRITE(6,'(A,10I3)') '  GAS spaces included           ',
     &               (ICMBSPC(II,ISPC),II=1,LCMBSPC(ISPC))
        WRITE(6,*)
        WRITE(6,*) ' Occupation constraints: '  
        WRITE(6,*) '========================= '
        WRITE(6,*)  
        WRITE(6,*)
        DO JJGASSPC = 1, LCMBSPC(ISPC)
         JGASSPC = ICMBSPC(JJGASSPC,ISPC)
        WRITE(6,*)
     &  ' Gas space  Min acc. occupation Max acc. occupation '
        WRITE(6,*)
     &  ' ================================================== '
        DO IGAS = 1, NGAS
          WRITE(6,'(3X,I2,13X,I3,16X,I3)') IGAS,
     &     IGSOCCX(IGAS,1,JGASSPC),IGSOCCX(IGAS,2,JGASSPC) 
        END DO
        END DO
*
       END IF
* 
      IF(NOCSF.EQ.0.AND.ICNFBAT.EQ.2) THEN
*. Test generate all CNF info
        CALL TEST_CNF_INFO_FOR_OCCLS
      END IF
*
      IF(IPRNT.GT.1) WRITE(6,*)
      NDET = XISPSM(ISM,ISPC)
*. Largest number of dets, any symmetry
      NDET_MAX = 0
      DO LSM = 1, NSMOB
        LLDET = XISPSM(LSM,ISPC)
        NDET_MAX = MAX(NDET_MAX, LLDET)
      END DO
      WRITE(6,*) ' NDET_MAX, NDET = ', NDET_MAX, NDET
*
      NCCM_STRING = XISPSM(ISM,ISPC)
      NSCM_STRIN = NCCM_STRING
      NEL = NELCI(ISPC)
      IF(IPRNT.GT.1)
     &WRITE(6,*) ' Number of STRING determinants/combinations  ',NDET
      IF(NDET.EQ.0) THEN
       WRITE(6,*) ' The number of determinants/combinations is zero.'
       WRITE(6,*) ' I am sure that fascinating discussions about '
       WRITE(6,*) ' the energy of such a wave function exists, '
       WRITE(6,*) ' but I am just a dumb program, so I will stop'
       WRITE(6,*)
       WRITE(6,*) ' GASCI: Vanishing number of parameters '
       STOP       ' GASCI: Vanishing number of parameters '
      END IF
*.Transfer to CANDS
      ICSM = ISM
      ISSM = ISM
      ICSPC = ISPC
      ISSPC = ISPC
*. Complete operator 
      I12 = 2
*. Info in A and B strings
      IATP = 1
      IBTP = 2
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
      IB_A = IBSPGPFTP(IATP)
      IB_B = IBSPGPFTP(IBTP)
*
      IF(NOCSF.EQ.1) THEN
        NVAR = NDET
        NVAR_MAX = NDET_MAX
      ELSE
        NCONF = NCONF_PER_SYM_GN(ISM, ISPC)
        NCSF =  NCSF_PER_SYM_GN(ISM, ISPC)
        NCCM_CONF = NCM_PER_SYM_GN(ISM, ISPC)
        NSCM_CONF = NCCM_CONF
        NDET = NSD_PER_SYM_GN(ISM, ISPC)
        NVAR = NCSF
*. There is pt only the single symmetry for CSFs, so
        NVAR_MAX = NCSF
        WRITE(6,'(A,2(2X,I9))')
     &  ' Number of CONF determinants/combinations in S, C',
     &    NCCM_CONF, NSCM_CONF
C?      WRITE(6,*) ' NCONF, NCSF, NSD = ', NCONF, NCSF, NSD
      END  IF
*
      NCVAR = NVAR
      NSVAR = NVAR
*
      IF(IPRNT.GE.5) WRITE(6,*) '  NVAR in GASCI ', NVAR
*. Allocate memory for diagonalization
*
*. Block for storing complete or partial CI-vector
      IF(ISIMSYM.EQ.1.OR.ICISTR.EQ.2) THEN
        LBLOCK = MXSOOB_AS
      ELSE
        LBLOCK = MXSOOB
      END IF
      IF(NOCSF.EQ.0.OR.ICNFBAT.EQ.-2) THEN
CERR    LBLOCK  = MAX(NSD_FOR_OCCLS_MAX,LBLOCK)
        LBLOCK  = MAX(N_SDAB_PER_OCCLS_MAX,LBLOCK)
      END IF
      LBLOCK = MAX(LBLOCK,LCSBLK)
      IF(NTEST.GE.1000) WRITE(6,*) ' TEST: LBLOCK = ', LBLOCK
* 
*. Information about block structure- needed by new PICO2 routine.
*. Memory for partitioning of C vector
      ICOMP = 0
      ILTEST = -3006
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' KCIOIO, KCBLTP, KCIOCCLS_ACT, KCLBT = ',
     &               KCIOIO, KCBLTP, KCIOCCLS_ACT, KCLBT 
        WRITE(6,*) ' KCLEBT, KCLBLK, KCI1BT, KCIBT = ',
     &               KCLEBT, KCLBLK, KCI1BT, KCIBT
        WRITE(6,*) ' KCNOCCLS_BAT, KCIBOCCLS_BAT = ',
     &               KCNOCCLS_BAT, KCIBOCCLS_BAT 
      END IF
*
      CALL Z_BLKFO_FOR_CISPACE(ISPC,ISM,LBLOCK,ICOMP,
     &     IPRNT,NCBLOCK,NCBATCH,
     &     int_mb(KCIOIO),int_mb(KCBLTP),NCOCCLS_ACT,
     &     dbl_mb(KCIOCCLS_ACT),
     &     int_mb(KCLBT),int_mb(KCLEBT),int_mb(KCLBLK),int_mb(KCI1BT),
     &     int_mb(KCIBT),
     &     int_mb(KCNOCCLS_BAT),int_mb(KCIBOCCLS_BAT),ILTEST)
C?    WRITE(6,*) ' WORK(KCI1BT)(1) '
C?    CALL IWRTMA(WORK(KCI1BT),1,1,1,1)
*
*. And for the Sigma vector
      CALL Z_BLKFO_FOR_CISPACE(ISPC,ISM,LBLOCK,ICOMP,
     &     IPRNT,NSBLOCK,NSBATCH,
     &     dbl_mb(KSIOIO),int_mb(KSBLTP),NSOCCLS_ACT,
     &     dbl_mb(KSIOCCLS_ACT),
     &     int_mb(KSLBT),int_mb(KSLEBT),int_mb(KSLBLK),int_mb(KSI1BT),
     &     int_mb(KSIBT),
     &     int_mb(KSNOCCLS_BAT),int_mb(KSIBOCCLS_BAT),ILTEST)
*. Number of BLOCKS
      NBLOCK = NCBLOCK
      NBATCH = NCBATCH
      IF(IPRNT.GT.1) THEN 
         WRITE(6,'(A,I9)') ' Number of blocks ', NBLOCK
         WRITE(6,'(A,I9)') ' Number of batches ', NBATCH
      END IF
C?    WRITE(6,*) ' TEST: NCBATCH, NSBATCH = ', NCBATCH, NSBATCH
*
* ===========================
* Info on configurations, etc
* ===========================
*
      IF(NOCSF.EQ.0.AND.ICNFBAT.EQ.1) THEN
*
*. Generate the configurations and order them according
*. to number of open orbitals
*
        CALL GEN_CONF_FOR_CISPC(dbl_mb(KCIOCCLS_ACT),NCOCCLS_ACT,ISM,
     &       int_mb(KIOCCLS))
*. The output are delivered in pointers set in GEN_CONF_FOR_CISPC
*         Occupations: KICONF_OCC(ISYM)
*         Reorder arrays: KICONF_REO(1) (independent of symmetry)
*. And then the reordering of the SD's
*. The reordering arrays for the SD's are stored in KSDREO_I((ISM)
C       WRITE(6,*) ' KSDREO_I(ISM) = ', KSDREO_I(ISM)
        CALL CNFORD_GAS(dbl_mb(KCIOCCLS_ACT),NCOCCLS_ACT,
     &       int_mb(KIOCCLS),ISM,PSSIGN,
     &       IPRCSF,int_mb(KICONF_OCC(ISM)),int_mb(KICONF_REO(1)),
     &       int_mb(KSDREO_I(ISM)),WORK(KSDREO_S(ISM)),
     &       int_mb(KCBLTP),int_mb(KCIBT),NCBLOCK)
C?      WRITE(6,*) ' IBLTP after CNFORD '
C?      CALL IWRTMA(WORK(KCBLTP),1,4,1,4)
      END IF ! all Conformation should be set up
*
      MXSTBL0 = MXNSTR           
*. type of alpha and beta strings
      IATP = 1              
      IBTP = 2             
*. alpha and beta strings with an electron removed
      IATPM1 = 3 
      IBTPM1 = 4
*. alpha and beta strings with two electrons removed
      IATPM2 = 5 
      IBTPM2 = 6
*
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
*. Largest number of strings of given symmetry and type
      MAXA = 0
      IF(NAEL.GE.1) THEN
        MAXA1 = IMNMX(int_mb(KNSTSO(IATPM1)),NSMST*NOCTYP(IATPM1),2)
C?      WRITE(6,*) ' MAXA1 1', MAXA1
        MAXA = MAX(MAXA,MAXA1)
      END IF
      IF(NAEL.GE.2) THEN
        MAXA1 = IMNMX(int_mb(KNSTSO(IATPM2)),NSMST*NOCTYP(IATPM2),2)
        MAXA = MAX(MAXA,MAXA1)
      END IF
      MAXB = 0
      IF(NBEL.GE.1) THEN
        MAXB1 = IMNMX(int_mb(KNSTSO(IBTPM1)),NSMST*NOCTYP(IBTPM1),2)
        MAXB = MAX(MAXB,MAXB1)
      END IF
      IF(NBEL.GE.2) THEN
        MAXB1 = IMNMX(int_mb(KNSTSO(IBTPM2)),NSMST*NOCTYP(IBTPM2),2)
        MAXB = MAX(MAXB,MAXB1)
      END IF
      MXSTBL = MAX(MAXA,MAXB,MXSTBL0)
      IF(IPRCIX.GE.2 ) WRITE(6,*)
     &' Largest block of strings with given symmetry and type',MXSTBL
*. Largest number of resolution strings and spectator strings
*  that can be treated simultaneously
      MAXI = MIN( MXINKA,MXSTBL)
      MAXK = MIN( MXINKA,MXSTBL)
*.scratch space for projected matrices and a CI block
*
*. Scratch space for CJKAIB resolution matrices
*. Size of C(Ka,Jb,j),C(Ka,KB,ij)  resolution matrices
      IOCTPA = IBSPGPFTP(IATP)
      IOCTPB = IBSPGPFTP(IBTP)
      CALL MXRESCPH(int_mb(KCIOIO),IOCTPA,IOCTPB,NOCTPA,NOCTPB,
     &            NSMST,NSTFSMSPGP,MXPNSMST,
     &            NSMOB,MXPNGAS,NGAS,NOBPTS,IPRCIX,MAXK,
     &            NELFSPGP,
     &            MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL,MXADKBLK,
     &            IPHGAS,NHLFSPGP,MNHL,IADVICE,MXCJ_ALLSYM,
     &            MXADKBLK_AS,MX_NSPII)
      IF(IPRCIX.GE.2) THEN
        WRITE(6,*) 'GASCI  : MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL',
     &                       MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL
        WRITE(6,*) ' MXADKBLK ,MXADKBLK_AS', MXADKBLK, MXADKBLK_AS
      END IF
      IF(ISIMSYM.EQ.1) THEN 
        MXCJ = MAX(MXCJ_ALLSYM,MX_NSPII)
        MXADKBLK_AS = MXADKBLK
      END IF
*. Using hardwired routines, MXCIJAB also used
      LSCR2 = MAX(MXCJ,MXCIJA,MXCIJB,MXCIJAB,MX_NSPII)
      IF(IPRCIX.GE.2) 
     &WRITE(6,*) ' Space for two resolution matrices ',2*LSCR2
      LSCR12 = MAX(LBLOCK,2*LSCR2)  
CBERT VECTOR DOUBLES!!! LOCAL
      CALL MEMMAN(KVEC3,LSCR12,'ADDL  ',2,'KC2   ')
      IF(IPRCIX.GE.3) WRITE(6,'(A,3(2X, I9))')
     &  'MXCJ,MXCIJA,MXCIJB,MXCIJAB = ',
     &   MXCJ,MXCIJA,MXCIJB,MXCIJAB
*
CBERT VECTOR DOUBLES!!! LOCAL
      CALL MEMMAN(KVEC1,LBLOCK,'ADDL  ',2,'VEC1  ')
      CALL MEMMAN(KVEC2,LBLOCK,'ADDL  ',2,'VEC2  ')
      KVEC1P = KVEC1
      KVEC2P = KVEC2
* 
      IF(ICISTR.EQ.1) THEN
*. Allocate space for two complete vectors of variables
        if ( .not. ga_create( mt_dbl, nvar_max, 1, 'CONVC1',
     $     0,0, KCOMVEC1) ) call errquit(
        if ( .not. ga_create( mt_dbl, nvar_max, 1, 'CONVC2',
     $     0,0, KCOMVEC2) ) call errquit(
     $     'GASCI: ga_create kcomvec2 failed', nvar_max , GA_ERR)
CNW    CALL MEMMAN(KCOMVEC1,NVAR_MAX,'ADDL  ',2,'COMVC1')
CNW    CALL MEMMAN(KCOMVEC2,NVAR_MAX,'ADDL  ',2,'COMVC2')
      ENDIF
*. Pointers to vectors holding complete or partial vector over variables
      IF(ICISTR.EQ.1) THEN
       KPVEC1 = KCOMVEC1
       KPVEC2 = KCOMVEC2
      ELSE
       KPVEC1 = KVEC1
       KPVEC2 = KVEC2
      END IF
      IF(NOCSF.EQ.0) THEN
       IF(ICNFBAT.EQ.1) THEN
*. For CSF's: Two vectors over CM's of Strings, all symmetries for
*generality
         CALL MEMMAN(KCOMVEC1_SD,NDET_MAX,'ADDL  ',2,'CMVC1D')
         CALL MEMMAN(KCOMVEC2_SD,NDET_MAX,'ADDL  ',2,'CMVC2D')
       ELSE
*. Memory for two blocks of combinations
         CALL MEMMAN(
     &   KCOMVEC1_SD,N_CMAB_PER_OCCLS_MAX,'ADDL  ',2,'CMVC1D')
         CALL MEMMAN(
     &   KCOMVEC2_SD,N_CMAB_PER_OCCLS_MAX,'ADDL  ',2,'CMVC2D')
       END IF
      END IF
* 
      IF(I_DO_COMHAM.EQ.1) THEN
         WRITE(6,*) ' Complete Hamiltonian matrix will be constructed '
         CALL MEMMAN(KLHMAT,NVAR*NVAR,'ADDL  ',2,'HMAT  ')
C COMHAM(H,NVAR,NBLOCK,LBLOCK,VEC1,VEC2)
         ECOREL = 0.0D0
         IF(NOCSF.EQ.1) THEN
*. Determinants: TTSS blocks
           CALL COMHAM(WORK(KLHMAT),NVAR,NBLOCK,int_mb(KCLBLK),
     &                 KPVEC1,KPVEC2,ECOREL)
         ELSE
*. CSFs: occlass blocks
*. Number of CSFs per occlass
           CALL MEMMAN(KLNCS_FOR_OCCLS,NCOCCLS_ACT,'ADDL  ',1,'NCS_OC')
           CALL GET_NCSF_PER_OCCLS_FOR_CISPACE(ISM,dbl_mb(KCIOCCLS_ACT),
     &          NCOCCLS_ACT,int_mb(KNCS_FOR_OCCLS),
     &          int_mb(KLNCS_FOR_OCCLS))
C          GET_NCSF_PER_OCCLS_FOR_CISPACE(ISYM,IOCCLS_ACT,
C    &           NOCCLS_ACT,NCS_FOR_OCCLS,NCS_FOR_OCCLS_ACT)
           CALL COMHAM(WORK(KLHMAT),NVAR,NCOCCLS_ACT,
     &          int_mb(KLNCS_FOR_OCCLS),KPVEC1,KPVEC2,ECOREL)
         END IF
         STOP ' Enforced stop after COMHAM'
      END IF
*
      IF(ICISTR.EQ.1) THEN
        LBLK = NVAR
      ELSE
        LBLK = - 1
      END IF
*
*. CI diagonal - if required
* 
      IF(NOCSF.EQ.0) THEN
        KPDIA_SD = KCOMVEC1_SD
      ELSE
        KPDIA_SD = KPVEC1
      END IF
*
      I_DO_PRECOND = 1
      IF(IDIAG.EQ.2.AND.IRESTR.EQ.1) I_DO_PRECOND = 0
      IF(I_DO_PRECOND.EQ.1) THEN
*. Calculate determinant diagonal if required
       IF(NOCSF.EQ.1.OR.(NOCSF.EQ.0.AND.IH0_CSF.EQ.1)) THEN
        ECOREP = 0.0D0
        IF(ICISTR.GE.2) CALL REWINO(LUDIA)
        I12 = 2
        IUSE_EXP = 1
        IF(IUSE_EXP.EQ.1) THEN
          SHIFT = ECORE
        ELSE
          SHIFT = ECORE_ORIG-ECORE
CBERT DOUBLE INTEGRALS
          CALL SWAPVE(WORK(KINT1),WORK(KINT1O),NINT1)
        END IF
*
C?      WRITE(6,*) ' IBLTP  before call to GASDIAT'
C?      CALL IWRTMA(WORK(KCBLTP),1,4,1,4)
*
C       CALL GASDIAT(WORK(KPVEC1),LUDIA,SHIFT,ICISTR,I12,
        CALL GASDIAT(KPDIA_SD,LUDIA,SHIFT,ICISTR,I12,
     &               int_mb(KCBLTP),NBLOCK,int_mb(KCIBT),IUSE_EXP)
C?      WRITE(6,*) ' WORK(KPDIA_SD) after GASDIAT =', WORK(KPDIA_SD)
        IF(IUSE_EXP.NE.1) THEN
          CALL SWAPVE(WORK(KINT1),WORK(KINT1O),NINT1)
        END IF
       END IF ! determinant diagonal should be calculated
*
       IF(NOCSF.EQ.0.AND.IH0_CSF.EQ.1) THEN
*. Average of determinant diagonal in each conf
C?       WRITE(6,*) ' IH0_CSF = ', IH0_CSF
         CALL CSDIAG(KPVEC2,WORK(KPDIA_SD),
     &        NCONF_PER_OPEN(1,ISM),MAXOP,ISM,
     &        int_mb(KSDREO_I(ISM)),NPCMCNF,NPCSCNF,IPRCSF,
     &        ICNFBAT,NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),
     &        int_mb(KCLBT),LUDIA,LUSC52)
C        CSDIAG(CSFDIA,DETDIA,NCNFTP,MAXOP,ISM,
C    &         ICTSDT,NPCMCNF,NPCSCNF,IPRCSF,
C    &         ICNFBAT,NOCCLS_ACT,IOCCLS_ACT,
C    &         LBLOCK,LUDIA_DET,LUDIA_CSF)
*
         IF(ICNFBAT.EQ.1) THEN
           CALL COPVEC(KPVEC2,WORK(KPDIA_SD),NVAR)
         ELSE
           CALL COPVCD(LUSC52,LUDIA,KPVEC2,1,-1)
         END IF
       END IF
*
       IF(NOCSF.EQ.0.AND.IH0_CSF.GE.2) THEN
*. Use exact diagonal or blocks
C?       WRITE(6,*) ' Ecore before GET_CSF.... ', ECORE
C?       WRITE(6,*) ' KPDIA_SD = ', KPDIA_SD
         CALL GET_CSF_H_PRECOND(NCONF_PER_OPEN(1,ISM),
     &        int_mb(KICONF_OCC(ISM)),WORK(KPDIA_SD),ECORE,
     &        LUDIA,NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),ISM)
C        GET_CSF_H_PRECOND(NCONF_FOR_OPEN,ICONF_OCC,H0,ECORE,
C    &   LUDIA,NOCCLS_ACT,IOCCLS_ACT),ISYM)
       END IF
*
       IF(ICISTR.EQ.1) THEN
         CALL REWINO(LUDIA)
C?     WRITE(6,*) ' WORK(KPDIA_SD) before TODSC =', WORK(KPDIA_SD)
         CALL TODSC(WORK(KPDIA_SD),NVAR,NVAR,LUDIA)
       END IF
*
       IF(IIUSEH0P.EQ.1) THEN
*. Is pt not reprogrammed for CSF's and ICISTR = 1
*. Diagonal with F
         CALL SWAPVE(WORK(KFI),WORK(KINT1O),NINT1)
         CALL GASDIAT(dbl_mb(KVEC1),LUSC52,SHIFT,ICISTR,1,
     &              int_mb(KCBLTP),NBLOCK,int_mb(KCIBT),IUSE_EXP)
         CALL SWAPVE(WORK(KFI),WORK(KINT1O),NINT1)
*. diag of (1-Lambda) F + Lambda H
         FAC1 = 1.0D0 - XLAMBDA
         FAC2 = XLAMBDA
         CALL VECSMD(dbl_mb(KVEC1),dbl_mb(KVEC2),FAC1,FAC2,
     &   LUSC52,LUDIA,LUSC53,1,LBLK)
         CALL COPVCD(LUSC53,LUDIA,dbl_mb(KVEC1),1,LBLK)
       END IF !IIUSEH0P.EQ.1
*
       IF(IPRCIX.GE.2) WRITE(6,*) ' Diagonal constructed  '
*
       IF(IPRNT.GE.100) THEN
        WRITE(6,*) ' Constructed diagonal '
        CALL WRTMAT(KPDIA_SD,1,NVAR,1,NVAR)
       END IF
*
      ELSE
         WRITE(6,*) ' Diagonal not calculated '
      END IF ! diagonal should be calculated
*
*
*. Explicit Hamiltonian
*
      NH0DIM = 0
      NP1 = 0
      NP2 = 0
      NQ = 0
      NPRDET = 0
*
      IF(ISBSPC_SEL.NE.0.AND.
     &   (.NOT.(ISBSPC_SEL.EQ.3.AND.ISBSPC_SPC.GT.ISPC))) THEN
        NSBDET = MXP1 + MXP2 + MXQ
        IF(NSBDET .GT. NVAR ) THEN
*. The allowed dimension of the  subspace is higher than the
*. total dimension.Reduce total dimension and P1,P2,Q dimensions
*. Reduce
           ISUB = NSBDET - NVAR
           NSBDET = NVAR
           WRITE(6,*) ' NOTE: NSBDET lowered to ... ',NSBDET
*
           ISUBQ = MIN(ISUB,MXQ)
           MXQ = MXQ - ISUBQ
           ISUB  = ISUB - ISUBQ
*
           ISUBP2 = MIN(ISUB,MXP2)
           MXP2 = MXP2 - ISUBP2
           ISUB = ISUB - ISUBP2
*
           ISUBP1 = MIN(ISUB,MXP1)
           MXP1 = MXP1 - ISUBP1
           ISUB = ISUB - ISUBP1
*
           WRITE(6,'(/A,3I5)')
     &     ' Modified MXP1,MXP2,MXQ ',MXP1,MXP2,MXQ
        END IF
*
* Obtain the addresses and dimension of the subspace
*
C       GET_SUBSPC_PRECOND_SPC(ISPC,ISM,ISEL,NSEL,
C     &           CBLK)
        KLH0_SUBDT = KH0
        CALL GET_SUBSPC_PRECOND_SPC(ISPC,ISM,WORK(KLH0_SUBDT),NH0DIM,
     &       dbl_mb(KVEC1))
        NPRDET = NH0DIM
        NP1 = NPRDET
        NP2 = 0
        NQ = 0
*
*. Obtain and diagonalize subsspace Hamiltonian
* 
*
*. Storage of H0, so it can transferred as a single 
*. array, H0. Pt assuming only a P1 precinditioner
        KLH0_MAT = KLH0_SUBDT + NH0DIM
        KLH0_EIGVAL = KLH0_MAT + NP1*(NP1+1)/2
        KLH0_EIGVEC = KLH0_EIGVAL + NP1
*
C       GET_SUBSPC_PRECOND_MAT(ISPC,ISM,H0,ISEL,NSEL,
C    &           EIGVAL, EIGVEC)
* (NP1, NP2, NQ transferred through common) CRUN
        CALL GET_SUBSPC_PRECOND_MAT(ISPC,ISM,WORK(KLH0_MAT),
     &       WORK(KLH0_SUBDT),NH0DIM,WORK(KLH0_EIGVAL),
     &       WORK(KLH0_EIGVEC))
      END IF
*
*. Transfer control to optimization routine
*
      MINST = 1
      IF(IRESTR.EQ.0) THEN
        INICI = 0
      ELSE
        INICI = -1
      END IF
*
      IF(ICISTR.EQ.1) THEN
        LBLK = NVAR
      ELSE
        LBLK = - 1
      END IF
*. This is a CI and not a perturbation calculation 
*. ( what about B-W perturbation theory ???? ) 
      IPERTOP = 0
*
* Space for class selection
*
      IF(ICLSSEL.EQ.1) THEN
*. Contribution to Energy and C per base CI space
*. P.S : BASECI space for a class : CI space where class is first allowed
        NOCCLS = NOCCLS_MAX
        CALL MEMMAN(KLEBASC,NOCCLS,'ADDL  ',2,'EBASC ')
        CALL MEMMAN(KLCBASC,NOCCLS,'ADDL  ',2,'CBASC ')
*. alphasupergroup, betasupergroup=> class
        CALL MEMMAN(KLSPSPCL,NOCTPA*NOCTPB,'ADDL  ',1,'SPSPCL')
        CALL SPSPCLS(int_mb(KLSPSPCL),int_mb(KIOCCLS),NOCCLS)
*. Class of each block
        CALL MEMMAN(KLBLKCLS,NBLOCK,'ADDL  ',1,'BLKCLS')
        CALL MEMMAN(KLCLSL,NOCCLS,'ADDL  ',1,'CLSL  ')
        CALL MEMMAN(KLCLSLR,NOCCLS,'ADDL  ',2,'CLSL_R')
        CALL BLKCLS(int_mb(KCIBT),NBLOCK,int_mb(KLBLKCLS),
     &              int_mb(KLSPSPCL),
     &              NOCCLS,int_mb(KLCLSL),NOCTPA,NOCTPB,dbl_mb(KLCLSLR))
*. Allocate space for additinal arrays used for class selection
        CALL MEMMAN(KLCLSC,NOCCLS,'ADDL  ',2,'CLSC  ')
        CALL MEMMAN(KLCLSE,NOCCLS,'ADDL  ',2,'CLSE  ')
        CALL MEMMAN(KLCLSCT,NOCCLS,'ADDL  ',2,'CLSCT ')
        CALL MEMMAN(KLCLSET,NOCCLS,'ADDL  ',2,'CLSET ')
        CALL MEMMAN(KLCLSA,NOCCLS,'ADDL  ',2,'CLSA  ')
        CALL MEMMAN(KLCLSA2,NOCCLS,'ADDL  ',2,'CLSA2 ')
        CALL MEMMAN(KLBLKA,NBLOCK,'ADDL  ',1,'BLKA  ')
        CALL MEMMAN(KLCLSD,NOCCLS,'ADDL  ',2,'CLSDE ')
        CALL MEMMAN(KLCLSDT,NOCCLS,'ADDL  ',2,'CLSDET')
        CALL MEMMAN(KLCLSG,NOCCLS,'ADDL  ',2,'CLSDE ')
        CALL MEMMAN(KLCLSGT,NOCCLS,'ADDL  ',2,'CLSDET')
      END IF


      IF(MULSPC.EQ.1.AND.ISPC.GE.IFMULSPC) THEN
        MULSPCA = 1
      ELSE
        MULSPCA = 0
      END IF
*
      IF(IDIAG.EQ.1) THEN
        EADD = 0.0D0
      ELSE
       EADD = ECORE
      END IF
*
      EROOT(1) = -2810
*. The two vectors that are sent through the machinary are KPVEC1, KPVEC2, i.e. full vectors
* parameters or blocks. If CSF's are in question, it is the vectors over parameters.
C?    WRITE(6,*) ' Before CIEIG5, LUC = ', LUC
      MPORENP_E = 0
C?    WRITE(6,*) ' KCI1BT, WORK(KCI1BT)(2) ', KCI1BT
C?    CALL IWRTMA(WORK(KCI1BT),1,1,1,1)
*
      IF(ISBSPPR.GT.0.AND.ISBSPPR_INI.GE.ISPC) THEN
        ISBSPPR_ACT = ISBSPPR
      ELSE
        ISBSPPR_ACT = 0
      END IF
*. An additional file
      CALL FILEMAN_MINI(LU8,'ASSIGN')
      ILAST = -3006
      CALL CIEIG5(MV7,INICI,EROOT,KPVEC1,KPVEC2,
     & MINST,LUDIA,LUC,LUHC,LUSC1,LUSC2,LUSC3,LUSC34,LUSC35,LU8,
     & NVAR,NBLK,NROOT,MXCIV,MAXIT,LUCIVI,IPRNT,WORK(KLH0_EIGVEC),
     & NPRDET,WORK(KH0),WORK(KLH0_SUBDT),
     & NP1,NP2,NQ,WORK(KH0SCR),EADD,ICISTR,LBLK,
     & IDIAG,dbl_mb(KVEC3),THRES_E,NBATCH,
     & int_mb(KCLBT),int_mb(KCLEBT),int_mb(KCLBLK),int_mb(KCI1BT),
     & int_mb(KCIBT),
     & int_mb(KSLBT),int_mb(KSLEBT),int_mb(KSLBLK),int_mb(KSI1BT),
     & int_mb(KSIBT),
     & INIDEG,E_THRE,C_THRE,
     & E_CONV,C_CONV,ICLSSEL,int_mb(KLBLKCLS),NOCCLS_MAX,
     & dbl_mb(KLCLSC),dbl_mb(KLCLSE),dbl_mb(KLCLSCT),dbl_mb(KLCLSET),
     & dbl_mb(KLCLSA),int_mb(KLCLSL),dbl_mb(KLCLSLR),int_mb(KLBLKA),
     & dbl_mb(KLCLSD),dbl_mb(KLCLSDT),dbl_mb(KLCLSG),dbl_mb(KLCLSGT),
     & ISKIPEI,int_mb(KC2B),dbl_mb(KLCLSA2),
     & LBLOCK,IROOT_SEL,int_mb(KBASSPC),dbl_mb(KLEBASC),
     & dbl_mb(KLCBASC),NCMBSPC,MULSPCA,IPAT,LPAT,ISPC,NCNV_RT,
     & IPRECOND,IIUSEH0P,MPORENP_E,RNRM,CONV_F,ISBSPPR_ACT,
     & ILAST)
*
      CALL FILEMAN_MINI(LU8,'FREE  ')
      EREF = EROOT(NROOT)
      ERROR_NORM_FINAL = RNRM(NROOT)
C?    WRITE(6,*) ' ERROR_NORM_FINAL on return from CIEIG5',
C?   &ERROR_NORM_FINAL
C?    WRITE(6,*) ' Memcheck after CIEIG5'
C?    CALL MEMCHK
*
*. Super-symmetry if required - done without density matrices
*. ===========================
*
      IF(I_DO_LZ2.EQ.1) THEN
*
       CALL REWINO(LUC)
       DO JROOT = 1, NROOT
        CALL FRMDSC(KPVEC1,NVAR,-1,LUC,IMZERO,IAMPACK)
        CALL EXP_LZ2(KPVEC1,KPVEC2,RLZEFF,RL2EFF,0)
        dbl_mb(KLZEXP-1+JROOT) = RLZEFF
        dbl_mb(KL2EXP-1+JROOT) = RL2EFF
       END DO ! loop over roots
       IF(NTEST.GE.00) THEN
        WRITE(6,*) ' The Lz and L(L+1) arrays:'
        WRITE(6,*)
        CALL WRTMAT(dbl_mb(KLZEXP),1,NROOT,1,NROOT)
        WRITE(6,*)
        CALL WRTMAT(dbl_mb(KL2EXP),1,NROOT,1,NROOT)
       END IF
      END IF ! supersymmetry should be calculated
*. Analyze density and CI-expansion for each ROOT
      CALL REWINO(LUC)
      DO JROOT = 1, NROOT
        IF(IPRNT.GT.1) THEN
        WRITE(6,*)
        WRITE(6,'(1H ,A,I3)')
     &  ' **************************************************'
        WRITE(6,'(1H ,A,I3)')
     &  ' Analysis of Density and occupation for ROOT = ',JROOT
        WRITE(6,'(1H ,A,I3)')
     &  ' **************************************************'
        WRITE(6,*)
        END IF
        IF(ICISTR.EQ.1) THEN
          CALL FRMDSC(KPVEC1,NVAR,-1,LUC,IMZERO,IAMPACK)
          CALL COPVEC(KPVEC1,KPVEC2,NVAR)
*
C?      WRITE(6,*) ' Testing: CI vector read in '
C?      CALL WRTMAT(WORK(KPVEC1),1,NVAR,1,NVAR)
*
          IF(IDENSI.EQ.1) THEN
            KRHO2 = 1
          END IF
          IF(IDENSI.GE.1)
     &    CALL DENSI2(IDENSI,WORK(KRHO1),WORK(KRHO2),
     &         KPVEC1,KPVEC2,0,0,EXPS2,ISPNDEN,WORK(KSRHO1),
     &         WORK(KRHO2AA),WORK(KRHO2AB),WORK(KRHO2BB),1)
          IF(IDENSI.EQ.2) THEN
*. Test calculation of energy
            CALL EN_FROM_DENS(ENERGY,2,0)
            WRITE(6,*) ' Energy from density ', ENERGY
          END IF
*
C?      IF(I_DO_LZ2.EQ.1) THEN
C?          CALL EXP_LZ2(KPVEC1,KPVEC2,RLZEFF,RL2EFF,0)
C?          WORK(KLZEXP-1+JROOT) = RLZEFF
C?          WORK(KL2EXP-1+JROOT) = RL2EFF
C?      END IF
*
          IF(IPRNT.GT.2) THEN
           IF(NOCSF.EQ.1) THEN
            CALL GASANA(KPVEC1,NBLOCK,int_mb(KCIBT),
     &                  int_mb(KCBLTP),LUC,ICISTR)
           ELSE
            THRES = 0.1D0
            MAXTRM = 100
            IOUT = 6
C              ANACSF(CIVEC,ICONF_OCC,NCONF_FOR_OPEN,IPROCS,THRES,MAXTRM,IOUT)
            IF(ICNFBAT.EQ.1) THEN
              CALL ANACSF(KPVEC1,int_mb(KICONF_OCC(ISM)),
     &                    NCONF_PER_OPEN(1,ISM),int_mb(KCFTP),
     &                    THRES,MAXTRM,IOUT)
            ELSE
              THRES = 0.1D0
              MAXTRM = 100
              IOUT = 6
              CALL ANACSF2(LUC,NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),ISM,
     &             KPVEC1,int_mb(KICONF_OCC(ISM)),
     &             NCONF_PER_OPEN(1,ISM),int_mb(KCFTP),THRES,
     &             MAXTRM,IOUT)
C    ANACSF2(LUC,NOCCLS_SPC,IOCCLS_SPC,
C   &           CIVEC,ICONF_OCC,NCONF_FOR_OPEN,IPROCS,THRES,
C   &           MAXTRM,IOUT)
              END IF ! CNF batch switch
           END IF! CSF switch
          END IF ! IPRNT > 2
        ELSE ! ICISTR switch
          IF(IDENSI.GE.1.OR.NROOT.GT.1) THEN
             CALL REWINO(LUSC1)
             CALL COPVCD(LUC,LUSC1,dbl_mb(KVEC1),0,LBLK)
             CALL COPVCD(LUSC1,LUHC,dbl_mb(KVEC1),1,LBLK)
          END IF
          IF(IDENSI.GE.1)
     &    CALL DENSI2(IDENSI,WORK(KRHO1),WORK(KRHO2),
     &          dbl_mb(KVEC1),dbl_mb(KVEC2),LUSC1,LUHC,EXPS2,
     &          ISPNDEN,WORK(KSRHO1),WORK(KRHO2AA),WORK(KRHO2AB),
     &          WORK(KRHO2BB),1 )
          IF(IPRNT.GT.1.AND.IDENSI.EQ.2)
     &    WRITE(6,*) ' Expectation value of S2', EXPS2
          IF(IDENSI.EQ.2) THEN
*. Test calculation of energy
            CALL EN_FROM_DENS(ENERGY,2,0)
            WRITE(6,*) ' Energy from density ', ENERGY
          END IF
*
C?        IF(I_DO_LZ2.EQ.1) THEN
C?            IF(NROOT.EQ.1) THEN
C?              LUCEFF = LUC
C?            ELSE
C?              LUCEFF = LUSC1
C?            END IF
C?            WRITE(6,*) ' EXP_LZ2 will be called '
C?            CALL EXP_LZ2(KPVEC1,KPVEC2,RLZEFF,RL2EFF,
C?   &              LUCEFF)
C?            WORK(KLZEXP-1+JROOT) = RLZEFF
C?            WORK(KL2EXP-1+JROOT) = RL2EFF
C?        END IF
*
          IF(ISPC.EQ.1.AND.JROOT.EQ.IH0ROOT) THEN
*. Calculate inactive Fock matrix for reference root in first space 
             CALL COPVEC(WORK(KINT1),WORK(KFIZ),NINT1)
             CALL FIFAM(WORK(KFIZ))
          END IF
*
          IF(IPRNT.GT.2) THEN
           IF(NROOT.GT.1) THEN
            CALL REWINO(LUSC1)
            IF(NOCSF.EQ.1) THEN
              CALL GASANA(dbl_mb(KVEC1),NBLOCK,int_mb(KCIBT),
     &                    int_mb(KCBLTP),LUSC1,ICISTR)
            ELSE
              IOUT = 6
              THRES = 0.1D0
              MAXTRM = 100
              CALL ANACSF2(LUC,NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),ISM,
     &             KPVEC1,int_mb(KICONF_OCC(ISM)),
     &             NCONF_PER_OPEN(1,ISM),int_mb(KCFTP),THRES,
     &             MAXTRM,IOUT)
            END IF
           ELSE
            CALL REWINO(LUC)
            IF(NOCSF.EQ.1) THEN
              CALL GASANA(dbl_mb(KVEC1),NBLOCK,int_mb(KCIBT),
     &                    int_mb(KCBLTP),LUC,ICISTR)
            ELSE
              IOUT = 6
              THRES = 0.1D0
              MAXTRM = 100
              CALL ANACSF2(LUC,NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),ISM,
     &             KPVEC1,int_mb(KICONF_OCC(ISM)),
     &             NCONF_PER_OPEN(1,ISM),int_mb(KCFTP),THRES,
     &             MAXTRM,IOUT)
            END IF !NOCSF switch
           END IF !Nroot switch
          END IF !IPRNT .gt. 2
C?        WRITE(6,*) ' Back from GASANA ' 
*
          IF(IPRNCIV.EQ.1) THEN
             WRITE(6,*)
             WRITE(6,*) ' ======================'
             WRITE(6,*) ' Complete CI expansion '
             WRITE(6,*) ' ======================'
             WRITE(6,*)
             IF(NROOT.EQ.1) THEN
               CALL WRTVCD(dbl_mb(KVEC1),LUC  ,1,LBLK)
             ELSE
               CALL WRTVCD(dbl_mb(KVEC1),LUSC1,1,LBLK)
             END IF
*         ^ End if print of CI vector
        END IF
*       ^ End of ICISTR switch
      END DO
*     ^ End of loop over roots
*
CM    IF(NTEST.GE.00.AND.I_DO_LZ2.EQ.1) THEN
CM      WRITE(6,*) ' The Lz and L(L+1) arrays:'
CM      WRITE(6,*)
CM      CALL WRTMAT(WORK(KLZEXP),1,NROOT,1,NROOT)
CM      WRITE(6,*)
CM      CALL WRTMAT(WORK(KL2EXP),1,NROOT,1,NROOT)
CM    END IF
*. Supersymmetry
CM    IF(I_DO_LZ2.EQ.1) THEN
CM      KLLSUPSYM = KLZEXP
CM      IF(CSUPSYM(1:6).EQ.'ATOMIC') THEN
CM        KLLSUPSYM = KL2EXP
CM      END IF
CM      XSUPSYM = WORK(KLLSUPSYM-1+NROOT)
CM      WRITE(6,*) ' Largest root and its supersymmetry ', 
CM   &  NROOT, XSUPSYM
CM    END IF
*
*. Select reference root
*
      ISROOT = NROOT
      IF((IRESTR.EQ.0.OR.ISEL_ONLY_INI.EQ.0).AND.
     &    ITG_SROOT.NE.NROOT) THEN
        WRITE(6,*) ' Before select'
        CALL SELECT_ROOT(NROOT,ISROOT)
C            SELECT_ROOT(NROOT_ACT,ISROOT)
        WRITE(6,*) ' Root selected as reference  ISROOT = ', ISROOT
*. Energy and residual for chosen root
        EREF = EROOT(ISROOT)
        ERROR_NORM_FINAL = RNRM(ISROOT)
*. Prepare LUC so it has root ISROOT as first root and remaining roots as following vectors
C     PREPARE_NEW_LUC(LUCIN,LUCOUT,NROOTIN,NROOTUT,ISROOT,
C    &           NVAR, ICISTR,ICOPY)
        CALL PREPARE_NEW_LUC(LUC,LUHC,NROOT,NROOT,ISROOT,
     &           NVAR,ICISTR,1,KPVEC1)
      END IF ! Root should be selected
*
*. Obtain densities etc for chosen reference root
*
      IF(ISROOT.NE.NROOT.AND.IDENSI.GE.1) THEN
          IF(ICISTR.EQ.1) THEN
            CALL REWINO(LUC)
            CALL FRMDSC(KPVEC1,NVAR,-1,LUC,IMZERO,IAMPACK)
            CALL COPVEC(KPVEC1,KPVEC2,NVAR)
            IF(IDENSI.EQ.1) THEN
              KRHO2 = 1
            END IF
            CALL DENSI2(IDENSI,WORK(KRHO1),WORK(KRHO2),
     &      KPVEC1,KPVEC2,0,0,EXPS2,ISPNDEN,WORK(KSRHO1),
     &      WORK(KRHO2AA),WORK(KRHO2AB),WORK(KRHO2BB),1)
          ELSE
            CALL REWINO(LUSC1)
            CALL REWINO(LUC)
            CALL COPVCD(LUC,LUSC1,dbl_mb(KVEC1),0,LBLK)
            CALL DENSI2(IDENSI,WORK(KRHO1),WORK(KRHO2),
     &          dbl_mb(KVEC1),dbl_mb(KVEC2),LUC,LUSC1,EXPS2,
     &          ISPNDEN,WORK(KSRHO1),WORK(KRHO2AA),WORK(KRHO2AB),
     &          WORK(KRHO2BB),1 )
          END IF ! ICISTR switch
      END IF ! densities should be calculated
*
*. Supersymmetry
CM    WRITE(6,*) ' Before second check of supersymmetry '
CM    IF(I_DO_LZ2.EQ.1.AND.INI_SROOT.NE.NROOT) THEN
CM      KLLSUPSYM = KLZEXP
CM      IF(CSUPSYM(1:6).EQ.'ATOMIC') THEN
CM        KLLSUPSYM = KL2EXP
CM      END IF
CM      XSUPSYM = WORK(KLLSUPSYM-1+ISROOT)
CM      WRITE(6,*) ' Supersymmetry of reference state ', XSUPSYM
CM    END IF
*
* Property section
*
      IF(NPROP.GT.0) THEN 
*. Properties between pairs of states
        DO IROOT1 = 1, NROOT
          DO IROOT2 = 1, IROOT1
            WRITE(6,*)
            WRITE(6,*) 
            WRITE(6,*)  
     &      ' ****************************************************'
            WRITE(6,*)
            IF(IROOT1.EQ.IROOT2) THEN
              WRITE(6,*) 
     &        '       Expectation values for ROOT = ', IROOT1
            ELSE
              WRITE(6,*) 
     &      '         Transition properties between roots',
     &                    IROOT1, IROOT2
            END IF
            WRITE(6,*)
            WRITE(6,*)  
     &      ' ****************************************************'
            WRITE(6,*)
            WRITE(6,*) 
*. Vector IROOT1 on  LUSC1, IROOT2 on LUHC
C                SKPVCD(LU,NVEC,SEGMNT,IREW,LBLK)
            CALL SKPVCD(LUC,IROOT1-1,dbl_mb(KVEC1),1,LBLK)
            CALL REWINO(LUSC1)
            CALL COPVCD(LUC,LUSC1,dbl_mb(KVEC1),0,LBLK)
*
            CALL SKPVCD(LUC,IROOT2-1,dbl_mb(KVEC1),1,LBLK)
            CALL REWINO(LUHC)
            CALL COPVCD(LUC,LUHC,dbl_mb(KVEC1),0,LBLK)
*  <IROOT1!E!IROOT2>
            IF(IDENSI.EQ.1) THEN
              KRHO2 = 1
            END IF
*. No reason to see on nat occ numbers again
            IPRDEN_SAVE = IPRDEN
            IPRDEN = 0
            XDUM = 0
            CALL DENSI2(1     ,WORK(KRHO1),WORK(KRHO2),
     &           dbl_mb(KVEC1),dbl_mb(KVEC2),LUSC1,LUHC,EXPS2,
     &           0,XDUM,XDUM,XDUM,XDUM,1)
            IPRDEN = IPRDEN_SAVE
            IF(IROOT1.EQ.IROOT2) THEN
              I_EXP_OR_TRA = 1
            ELSE
              I_EXP_OR_TRA = 2
            END IF
*. No relaxation terms
            IF(IROOT1.NE.IROOT2 .OR. IRELAX.EQ.0) THEN
              IRELAX_LOC = 0
              KLRESP_DEN = 1
            ELSE 
*. Include orbital relaxation: 1) Fock matrix 2) Response density
              WRITE(6,*) ' Construct fock matrix '
              CALL MEMMAN(KLFOO,NTOOB**2,'ADDL  ',2,'FOO   ')
              CALL FOCK_MAT(WORK(KLFOO),2)
C             WRITE(6,*) ' Fock matrix'
C             CALL APRBLM2(WORK(KLFOO),NTOOBS,NTOOBS,NSMOB,0)
              CALL MEMMAN(KLRESP_DEN,NTOOB**2,'ADDL  ',2,'RESP_D')
              IRELAX_LOC = 1
              CALL RESPDEN_FROM_F(WORK(KLFOO),WORK(KLRESP_DEN))
C                  RESPDEN_FROM_F(FOCK,RESPDEN)
            END IF
*
            IF(NPROP.GT.0) 
     &      CALL ONE_EL_PROP(I_EXP_OR_TRA,IRELAX_LOC,WORK(KLRESP_DEN))
*
C           IF(IROOT1.EQ.IROOT2.AND.I_DO_LZ2.EQ.1) THEN
C             CALL EXP_LZ2(WORK(KVEC1),WORK(KVEC2),RLZEFF,RL2EFF)
C           END IF
          END DO
        END DO
      END IF
*
*. For testing purposes: calculate energy and Fock matrix
*
       ITEST = 0
       IF(ITEST.EQ.1.AND.IDENSI.EQ.2) THEN
         WRITE(6,*) 'calling EN_FROM_DEN'
         CALL EN_FROM_DENS(ENERGY,2,0)
         WRITE(6,'(A,F24.12)') 
     &   ' Energy from EN_FROM_DEN',ENERGY 
         WRITE(6,*) ' Construct fock matrix '
         CALL MEMMAN(KLFOO,NTOOB**2,'ADDL  ',2,'FOO   ')
         CALL FOCK_MAT(WORK(KLFOO),2)
         WRITE(6,*) ' Fock matrix'
         CALL APRBLM2(WORK(KLFOO),NTOOBS,NTOOBS,NSMOB,0)
       END IF
*
*. Regenerate densities for reference root
*
C?     write(6,*) ' IRFROOT and NROOT ',IRFROOT, NROOT
       IF(IH0ROOT.NE.NROOT) THEN
*. Position vectors 
         CALL REWINO(LUC)
         DO JROOT = 1, IH0ROOT
           IF(ICISTR.EQ.1) THEN
             CALL FRMDSC(dbl_mb(KVEC1),NVAR,-1,LUC,IMZERO,IAMPACK)
           ELSE
             CALL REWINO(LUSC1)
             CALL COPVCD(LUC,LUSC1,dbl_mb(KVEC1),0,LBLK)
           END IF
         END DO
         IF(ICISTR.EQ.1) THEN
           CALL COPVEC(dbl_mb(KVEC1),dbl_mb(KVEC2),NVAR)
           CALL DENSI2(1,WORK(KRHO1),WORK(KRHO2),
     &          dbl_mb(KVEC1),dbl_mb(KVEC2),0,0,EXPS2,
     &          ISPNDEN,WORK(KSRHO1),WORK(KRHO2AA),WORK(KRHO2AB),
     &          WORK(KRHO2BB),1 )
         ELSE
           CALL REWINO(LUSC1)
           CALL COPVCD(LUSC1,LUSC2,dbl_mb(KVEC1),1,LBLK)
             CALL DENSI2(1,WORK(KRHO1),WORK(KRHO2),
     &            dbl_mb(KVEC1),dbl_mb(KVEC2),LUSC1,LUSC2,EXPS2,ISPNDEN,
     &            WORK(KSRHO1),WORK(KRHO2AA),WORK(KRHO2AB),
     &            WORK(KRHO2BB),1)
         END IF
       END IF
*
       IF(IEXTKOP.EQ.1) THEN
         WRITE(6,*) ' Control will be transferred to EXTKOP'
         CALL EXTKOP
       END IF
*
* CI respons for the final state
*
*. Note: DOES RESPONS for lowest state on LUC !
* make a copy of state of interest in general
      DO IRSPST = 1, NRSPST
        IRSPRT_CUR = IRSPRT(IRSPST)  
        ENER = EROOT(IRSPRT_CUR)
*. Active part of energy
C?      WRITE(6,*) 
C?   &  ' Control will be transferred to the response module'
C transfer state of interest to LUSC51
        WRITE(6,*) 'Response module called for root ', IRSPRT_CUR

        IF (IRSPRT_CUR.GT.1) THEN
           CALL SKPVCD(LUC,IRSPRT_CUR-1,dbl_mb(KVEC1),1,LBLK)
        ELSE 
          CALL REWINO(LUC)
        END IF
        CALL REWINO(LUSC51)
        CALL COPVCD(LUC,LUSC51,dbl_mb(KVEC1),0,LBLK)

C LUSC51 contains reference state
        CALL CI_RESPONS(LUHC,LUSC1,LUSC2,LUSC36,
     &                  LUSC38,LUSC39,LUSC40,LUSC51,LUDIA,
c     &                  LUSC38,LUSC39,LUSC40,LUC,LUDIA,
     &                  dbl_mb(KVEC1),dbl_mb(KVEC2),ENER)
c     &                  WORK(KVEC1),WORK(KVEC2),EREF)
C?      WRITE(6,*) ' Home from CI_RESPONS'
      END DO
*
      CALL MEMMAN(IDUMMY,IDUMMY,'FLUSM ',IDUM,'GASCI ')
      RETURN
      END
      SUBROUTINE HMATAPR_OLD(IASM,IATP,IBSM,IBTP,JASM,JATP,JBSM,JBTP,
     &           IAEL1,IBEL1,IAEL3,IBEL3,JAEL1,JBEL1,JAEL3,JBEL3,
     &           IAPRLEV)
*
* Determine at which level the Hamiltonian block
*
* <Iasm Iatp Ibsm Ibtp ! H ! Jasm Jatp Jbsm JBtp >
*
* should be calculated
*
*. IAPRLEV = -1 => No approximation
*. IAPRLEV = 0  => set block to zero
*. IAPRLEV = 1  => diagonal approximation
*. IAPRLEV = 2  => Use effective one-electronoperator
*
* Jeppe Olsen, Oct 1994
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'oper.inc'
C     COMMON/OPER/I12,IPERTOP,IAPR,MNRS1E,MXRS3E,IPART

*
      IF(IAPR.EQ.0) THEN
*. Full matrix => no approximations
        IAPRLEV = -1
      ELSE IF (IAPR.NE.0) THEN
*. RAS1,RAS3 checks
        IF(IAEL1+IBEL1.LT.MNRS1E.OR.
     &     IAEL3+IBEL3.GT.MXRS3E) THEN
*. I dets belongs to dets where approximations are allowed
           ILAPR = 1
        ELSE
*. Idets belongs to dets that are described exactly
           ILAPR = 0
        END IF 
*
        IF(JAEL1+JBEL1.LT.MNRS1E.OR.
     &     JAEL3+JBEL3.GT.MXRS3E) THEN
*. I dets belongs to dets where approximations are allowed
           IRAPR = 1
        ELSE
*. Idets belongs to dets that are described exactly
           IRAPR = 0
        END IF 
*
        IF(ILAPR.EQ.0.OR.IRAPR.EQ.0) THEN
*. No approximations
          IAPRLEV = -1
        ELSE
*. Diagonal block ?
          IF(IASM.EQ.JASM.AND.IATP.EQ.JATP.AND.
     &       IBSM.EQ.JBSM.AND.IBTP.EQ.JBTP) THEN
*. Yes:
             IF(IPART.EQ.2) THEN
*. Epstein- Nesbet partitioning
             IAPRLEV = +1
*. Moller -Plesset partitioning
             ELSE
               IAPRLEV = 2
             END IF
          ELSE
*. No  
             IAPRLEV = 0
          END IF
        END IF
      END IF
*
      NTEST = 000
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' HMATAPR reporting: '
        WRITE(6,*) ' Hamiltonian block in question '
        WRITE(6,'(A)') 
     &' IASM IATP IBSM IBTP JASM JATP JBSM JBTP '
        WRITE(6,'(1H ,8I5)')
     &  IASM,IATP,IBSM,IBTP,JASM,JATP,JBSM,JBTP  
        WRITE(6,'(A,4I4)') 'IAEL1 IBEL1 IAEL3 IBEL3 ',
     &                      IAEL1,IBEL1,IAEL3,IBEL3
        WRITE(6,'(A,4I4)') 'JAEL1 JBEL1 JAEL3 JBEL3 ',
     &                      JAEL1,JBEL1,JAEL3,JBEL3
        WRITE(6,'(A,2I4)') 'MNRS1E MXRS3E ', MNRS1E,MXRS3E
*
        WRITE(6,*) ' IAPRLEV = ', IAPRLEV
      END IF
*
      RETURN
      END 
      SUBROUTINE PERTCTL(ISM,ISPC,EREF,EFINAL)
*
* Master routine for perturbation calculations
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      EXTERNAL MV7
      INCLUDE 'cicisp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'cintfo.inc'
      
*. And defining perturbation operator
      INCLUDE 'oper.inc'
*
      INCLUDE 'csfbas.inc'
*. Common block for communicating with sigma
      INCLUDE 'cands.inc'
      INCLUDE 'cecore.inc'
*
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'PERTCT')


*
      WRITE(6,*) '**************************************'
      WRITE(6,*) '*                                    *'
      WRITE(6,*) '*   Perturbation calculation         *'
      WRITE(6,*) '*                                    *'
      WRITE(6,*) '**************************************'
*
      WRITE(6,*)
      WRITE(6,*) '  Largest order of correction vector ',NPERT
      WRITE(6,*)
      WRITE(6,*) ' ( IPART at start of PERTCTL ) ', IPART
      IF(IPART.EQ.1) THEN
        WRITE(6,*) ' Moller-Plesset Partitioning'
      ELSE IF(IPART.EQ.2) THEN
        WRITE(6,*) ' Epstein-Nesbet Partitioning'
      ELSE IF (IPART.EQ.3) THEN
        WRITE(6,*) ' One-body Hamiltonian read in '
      END IF
      IF(IE0AVEX.EQ.1) THEN
          WRITE(6,*)
     &  ' expectation value of H0 used as zero order energy '
      ELSE IF( IE0AVEX.EQ.2) THEN
          WRITE(6,*)
     &  ' Exact energy of reference used as zero order energy'
      ELSE IF( IE0AVEX.EQ.3) THEN
          WRITE(6,*)
     &  ' Read in energy is used as zero order energy'
      END IF
      WRITE(6,*) ' Root choosen as zero order state ', IRFROOT
*
*
*. 0: Initialization
*
      IF(NOCSF.EQ.0) THEN
        WRITE(6,*) ' Please turn off csf''s '
        STOP'NO CSF''s in PERTCTL !! '
      END IF
*
      NTEST = 10
      IPRNT = NTEST
      NTEST = MAX(NTEST,IPRNT)
      NDET = XISPSM(ISM,ISPC)
      NEL = NELCI(ISPC)
      WRITE(6,*) ' ISM ISPC ', ISM,ISPC
      WRITE(6,*) ' Number of determinants in internal space ',NDET
*.Transfer to CANDS
      ICSM = ISM
      ISSM = ISM
      ICSPC = ISPC
      ISSPC = ISPC
      WRITE(6,*) ' PERTCTL: ICSPC ISSSPC: ', ICSPC,ISSPC
      NVAR = NDET
      IF(IPRNT.GE.5)
     &WRITE(6,*) '  NVAR in REFCI ', NVAR
*. Arrays for S, V, H0 over correction vectors
      LENNY = (NPERT+1)*(NPERT+3)/2
      WRITE(6,*) ' LENNY ', LENNY
      CALL MEMMAN(KLSMAT ,LENNY,'ADDL  ',2,'LSMAT ')
      CALL MEMMAN(KLVMAT ,LENNY,'ADDL  ',2,'VSMAT ')
      CALL MEMMAN(KLH0MAT,LENNY,'ADDL  ',2,'H0SMAT')
*. Energy correction and scratch vector
      CALL MEMMAN(KLEN ,2*NPERT+2,'ADDL  ',2,'EN    ' )
      CALL MEMMAN(KLSCR,NPERT+1,'ADDL  ',2,'SCR   ' )
      
      
*. Allocate memory for diagonalization
      IF(ICISTR.EQ.1) THEN
        LBLOCK = NDET
      ELSE IF (ICISTR.EQ.2) THEN
        LBLOCK = MXSB
      ELSE IF (ICISTR.EQ.3) THEN
        LBLOCK = MXSOOB
      END IF
C?    WRITE(6,*) ' ICISTR = ', ICISTR
      LBLOCK = MAX(LCSBLK,LBLOCK)
      CALL MEMMAN(KVEC1,LBLOCK,'ADDL  ',2,'VEC1  ')
*. Vec2 will also be used as scratch in explicit hamiltonian generation
*. for CSF's
      IF(NOCSF.NE.0) THEN
        LBLOC2 = LBLOCK
      ELSE
        CALL LCNHCN(LSCR)
        LBLOC2 = MAX(LBLOCK,LSCR)
      END IF
      CALL MEMMAN(KVEC2,LBLOC2,'ADDL  ',2,'VEC2  ')
*. Sblock is used in general nowadays so, allocate an extra block
      I_USE_SBLOCK=1
      IF(I_USE_SBLOCK.EQ.1) THEN
*. Largest block of strings in zero order space
      MXSTBL0 = MXNSTR           
*. type of alpha and beta strings
      IATP = 1              
      IBTP = 2             
*. alpha and beta strings with an electron removed
      IATPM1 = 3 
      IBTPM1 = 4
*. alpha and beta strings with two electrons removed
      IATPM2 = 5 
      IBTPM2 = 6
*
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
*. Largest number of strings of given symmetry and type
      MAXA = 0
      IF(NAEL.GE.1) THEN
        MAXA1 = IMNMX(int_mb(KNSTSO(IATPM1)),NSMST*NOCTYP(IATPM1),2)
        MAXA = MAX(MAXA,MAXA1)
      END IF
      IF(NAEL.GE.2) THEN
        MAXA1 = IMNMX(int_mb(KNSTSO(IATPM2)),NSMST*NOCTYP(IATPM2),2)
        MAXA = MAX(MAXA,MAXA1)
      END IF
      MAXB = 0
      IF(NBEL.GE.1) THEN
        MAXB1 = IMNMX(int_mb(KNSTSO(IBTPM1)),NSMST*NOCTYP(IBTPM1),2)
        MAXB = MAX(MAXB,MAXB1)
      END IF
      IF(NBEL.GE.2) THEN
        MAXB1 = IMNMX(int_mb(KNSTSO(IBTPM2)),NSMST*NOCTYP(IBTPM2),2)
        MAXB = MAX(MAXB,MAXB1)
      END IF
      MXSTBL = MAX(MAXA,MAXB)
      IF(IPRCIX.GE.2 ) WRITE(6,*)
     &' Largest block of strings with given symmetry and type',MXSTBL
*. Largest number of resolution strings and spectator strings
*  that can be treated simultaneously
      MAXI = MIN( MXINKA,MXSTBL)
      MAXK = MIN( MXINKA,MXSTBL)
*.scratch space for projected matrices and a CI block
*
*. Scratch space for CJKAIB resolution matrices
*. Size of C(Ka,Jb,j),C(Ka,KB,ij)  resolution matrices
        IOCTPA = IBSPGPFTP(IATP)
        IOCTPB = IBSPGPFTP(IBTP)
*
        NOCTPA = NOCTYP(IATP)
        NOCTPB = NOCTYP(IBTP)
* 
        CALL MEMMAN(KLCIOIO,NOCTPA*NOCTPB,'ADDL  ',2,'CIOIO ')
        CALL IAIBCM(ISPC,dbl_mb(KLCIOIO))
        CALL MXRESCPH(dbl_mb(KLCIOIO),IOCTPA,IOCTPB,NOCTPA,NOCTPB,
     &              NSMST,NSTFSMSPGP,MXPNSMST,
     &              NSMOB,MXPNGAS,NGAS,NOBPTS,IPRCIX,MAXK,
     &              NELFSPGP,
     &              MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL,MXADKBLK,
     &              IPHGAS,NHLFSPGP,MNHL,IADVICE,MXCJ_ALLSYM,
     &              MXADKBLK_AS,MX_NSPII)
*
COLD    CALL MXRESC(WORK(KLCIOIO),IOCTPA,IOCTPB,NOCTPA,NOCTPB,
COLD &              NSMST,NSTFSMSPGP,MXPNSMST,
COLD &              NSMOB,MXPNGAS,NGAS,NOBPTS,IPRCIX,MAXK,
COLD &              NELFSPGP,
COLD &              MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL,MXADKBLK)
        IF(IPRCIX.GE.2) THEN
          WRITE(6,*) 'PERTCT: MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL',
     &                         MXCJ,MXCIJA,MXCIJB,MXCIJAB,MXSXBL
           WRITE(6,*) 'PERTCT: MXADKBLK ', MXADKBLK
        END IF
        LSCR2 = MAX(MXCJ,MXCIJA,MXCIJB,MXCIJAB,MX_NSPII)
        IF(IPRCIX.GE.2)
     &  WRITE(6,*) ' Space for resolution matrices ',LSCR2
        LSCR12 = MAX(LBLOCK,2*LSCR2)  
        CALL MEMMAN(KVEC3,LSCR12,'ADDL  ',2,'KC2   ')
      END IF
*
*. 1: Construct zero order operator: FI + FA
*
*. Copy root defining zero order operator to first vectors 
      CALL REWINO(LUHC)
      CALL REWINO(LUC)
      WRITE(6,*) ' Root used to define Zero order op ', IH0ROOT
      DO JROOT = 1, IH0ROOT
        CALL REWINO(LUSC36)
        CALL COPVCD(LUC,LUSC36,WORK(KVEC1),0,-1)
      END DO
      CALL COPVCD(LUSC36,LUHC,WORK(KVEC1),1,-1)
*. Construct corresponding one-body density matrix 
C     KRHO2 = 1
      XDUM = 0.0D0
      CALL DENSI2(1,WORK(KRHO1),WORK(KRHO2),WORK(KVEC1),WORK(KVEC2),
     &     LUHC,LUSC36,EXPS2,0,XDUM,XDUM,XDUM,XDUM,1)
*
*. Initialize with proper zero order root
*
      WRITE(6,*) ' After DENSI2, LUC LUSC36',LUC,LUSC36
      CALL REWINO(LUC)
      DO JROOT = 1, IRFROOT
        CALL REWINO(LUSC36)
        CALL COPVCD(LUC,LUSC36,WORK(KVEC1),0,-1)
      END DO
      CALL COPVCD(LUSC36,LUC,WORK(KVEC1),1,-1)
*
      LU18 = IGETUNIT(18)
*. Will MP operator be invoked
      IUSEMP = 0
      IF(NPTSPC.EQ.0) THEN
*. Use IPART
        IF(IPART.EQ.1) IUSEMP = 1
      ELSE
*. Check explicitly
        DO IISPC = 1, NPTSPC
         IF(IH0INSPC(IISPC).EQ.1.OR.IH0INSPC(IISPC).EQ.3
     &      .OR.IH0INSPC(IISPC).EQ.5) IUSEMP = 1
        END DO
      END IF
*
      WRITE(6,*) ' Testy, IUSEMP = ', IUSEMP
      IF(IUSEMP.EQ.1) THEN
        WRITE(6,*) ' Moller-Plesset operator will be used '
      ELSE
        WRITE(6,*) ' Moller-Plesset operator will not be used '
      END IF
*. Construct MP Hapr
      IF(IUSEMP.EQ.1) THEN
        CALL COPVEC(WORK(KINT1O),WORK(KFI),NINT1)
        CALL FIFAM(WORK(KFI))
        CALL COPVEC(WORK(KFI),WORK(KFIO),NINT1)
        WRITE(6,*) ' FI + FA matrix '
        CALL APRBLM2(WORK(KFI),NTOOBS,NTOOBS,NSMOB,1)
        ECORE_H = 0.0D0
        IF(IUSE_PH.EQ.1) THEN
         CALL FI(WORK(KFI),ECORE_H,0)
        END IF
*. Should a part of original one electron operator be
*  copied ( For mix exact Hamiltonian/Fock arroaches )
        IF(NH0EXSPC.NE.0) THEN
C             MIXHONE(H1,H2,NSMOB,NREPTP,IREPTP,NSMOB,NOBTP)
         CALL MIXHONE(WORK(KFI),WORK(KINT1),NH0EXSPC,IH0EXSPC,NGAS,
     &                NSMOB)
        END IF
      ELSE 
        CALL COPVEC(WORK(KINT1),WORK(KFI),NINT1)
        CALL COPVEC(WORK(KINT1O),WORK(KFIO),NINT1)
      END IF
      IF (IPART.EQ.3) THEN
*. Read in from file 18
        REWIND (LU18)
        CALL FRMDSC(WORK(KFI),NINT1,-1,LU18,IMZERO,IAMPACK)
        CALL COPVEC(WORK(KFI),WORK(KFIO),NINT1)
        ECORE_H = 0
        IF(IUSE_PH.EQ.1) THEN
         CALL FI(WORK(KFI),ECORE_H,0)
        END IF
        WRITE(6,*) ' H0 read in from LU18 '
        CALL APRBLM2(WORK(KFI),NTOOBS,NTOOBS,NSMOB,ISM)
*. Continue as mormal MP a piece of dirty code can never harm
        IPART = 1
        MPORENP = 1
      END IF 
*. Save H0 for future generations
      REWIND  LU18 
      CALL TODSC(WORK(KFI),NINT1,-1,LU18)
      REWIND LU18
       
*. No explicit construction of diagonal
      IDIDIA = 1
      IF(IDIDIA.EQ.0) THEN
*
*. 2: Diagonal with FI + FA
*
*. swap H and FI + FA
        IF(IPART.EQ.1) THEN  
          CALL SWAPVE(WORK(KFI),WORK(KINT1),NINT1)
          CALL SWAPVE(WORK(KFIO),WORK(KINT1O),NINT1)
        END IF
        IF(ICISTR.GE.2) CALL REWINO(LUDIA)
*. Transfer to COPER
        IPERTOP = 1
        IF(IPART.EQ.1) THEN
          I12 = 1
        ELSE
          I12 = 2
        END IF
        ECOREP = ECORE_H
        STOP ' GASDIAT call should be updated '
C       CALL GASDIAT(WORK(KVEC1),LUDIA,ECOREP,ICISTR,I12)
        IF(NOCSF.EQ.1.AND.ICISTR.EQ.1) THEN
          CALL REWINO(LUDIA)
          CALL TODSC(WORK(KVEC1),NVAR,-1,LUDIA)
C       ELSE IF(ICISTR.EQ.1.AND.NOCSF.EQ.0) THEN
C         CALL CSDIAG(WORK(KVEC2),WORK(KVEC1),NCNATS(1,ISM),NTYP,
C    &                WORK(KICTS(1)),NDPCNT,NCPCNT,0,
C    &                0,IDUM,IPRNT)
C         CALL REWINO(LUDIA)
C         CALL TODSC(WORK(KVEC2),NVAR,-1,LUDIA)
*. For transfer to H0CSF
C         CALL COPVEC(WORK(KVEC2),WORK(KVEC1),NVAR)
        END IF
*. swap H and FI + FA to get things in right place !
        IF(IPART.EQ.1)  THEN
          CALL SWAPVE(WORK(KFI),WORK(KINT1),NINT1)
          CALL SWAPVE(WORK(KFIO),WORK(KINT1O),NINT1)
        END IF
      END IF
*
* Transfer control to perturbation iterater
*
*. IS there a pert of Hamiltonian that is no diagonal
*. (requires solution of linear equations )
      IH0DIA = 1
      DO IISPC = 1, NPTSPC
        IF(IH0INSPC(IISPC).EQ.3.OR.IH0INSPC(IISPC).EQ.4.OR.
     &     IH0INSPC(IISPC).EQ.5) IH0DIA=0
      END DO
*
      IF(IH0DIA.EQ.0) THEN
        WRITE(6,*) ' Nondiagonal Approximate Hamiltonian '
      ELSE
        WRITE(6,*) ' Diagonal approximate Hamiltonian '
      END IF

*
* Nondiagonal form of perturbations: Currently indicated by
* operator type 3 and 4 
      IF(ICISTR.EQ.1) THEN
        LBLK = NVAR
      ELSE
        LBLK = - 1
      END IF
*
*. Transfer to COPER
*.  Perturbation matrix
       IPERTOP = 1
       IF(IPART.EQ.1) THEN
        I12 = 1
       ELSE
        I12 = 2
       END IF
*
      IF(IE0AVEX.EQ.3) THEN
        EREF = E0READ-ECORE
        WRITE(6,*) ' Zero order energy read in - ECORE ',EREF
      END IF
      CALL SIMPRT(LUC,LUSC36,LUHC,WORK(KLEN),WORK(KLSCR),
     &            NPERT,WORK(KVEC1),WORK(KVEC2),
     &            LUSC1,LUSC2,LBLK,IH0DIA,LUDIA,WORK(KLSMAT),
     &            WORK(KLVMAT),WORK(KLH0MAT),ECORE,ECORE_H,
     &            ECORE_HEX,EREF,IE0AVEX,LUSC39,EFINAL)
      WRITE(6,*) ' Testy, EFINAL in PERTCTL ', EFINAL
*. Analyze space spanned by zero order state and correction vectors
      I_CALL_PERT_SUBSPACE = 0
      IF(I_CALL_PERT_SUBSPACE.EQ.1)
     &CALL PERT_SUBSPACE(NPERT,WORK(KLH0MAT),
     &     WORK(KLVMAT),WORK(KLSMAT),ECORE)
*
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'PERTCT')
      RETURN
      END
      SUBROUTINE NATORB(RHO1,NSMOB,NTOPSM,NACPSM,NINPSM,
     &                  ISTOB,XNAT,RHO1SM,OCCNUM,
     &                  NACOB,SCR,IPRDEN)
*
* Obtain natural orbitals in symmetry blocks
*
* Jeppe Olsen, June 1994
*              Modification, Oct 94
*              Last modification, Feb. 1998 (reorder deg eigenvalues)
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION RHO1(NACOB,NACOB)
      DIMENSION ISTOB(*)
      DIMENSION NTOPSM(NSMOB), NACPSM(NSMOB),NINPSM(NSMOB)
*. Output
      DIMENSION RHO1SM(*),OCCNUM(*),XNAT(*)
*. Scratch ( Largest symmetry block )
      DIMENSION SCR(*)
*
      NTESTL = 0
      NTEST = MAX(NTESTL,IPRDEN)
*. To get rid of annoying and incorrect compiler warnings 
      IOBOFF = 0
      IMTOFF = 0
*. IOBOFF: Offset for active orbitals in symmetry order
      DO ISMOB = 1, NSMOB
        IF(ISMOB.EQ.1) THEN
          IOBOFF = NINPSM(1)+1
          IMTOFF = 1
        ELSE
          IOBOFF =
     &    IOBOFF + NTOPSM(ISMOB-1)-NINPSM(ISMOB-1)+NINPSM(ISMOB)
          IMTOFF = IMTOFF + NACPSM(ISMOB-1)**2
        END IF
        LOB = NACPSM(ISMOB)
*
*. Extract symmetry block of density matrix
*
        DO IOB = IOBOFF,IOBOFF + LOB-1
           DO JOB = IOBOFF,IOBOFF + LOB-1
*. Corresponding type indeces
             IOBP = ISTOB(IOB)
             JOBP = ISTOB(JOB)
             RHO1SM(IMTOFF-1+(JOB-IOBOFF)*LOB+IOB-IOBOFF+1)
     &     = RHO1(IOBP,JOBP)
           END DO
        END DO
*
        IF(NTEST.GE.2 ) THEN
          WRITE(6,*)
          WRITE(6,*) ' Density matrix for symmetry  = ', ISMOB
          WRITE(6,*) ' ======================================='
          WRITE(6,*)
          CALL WRTMAT(RHO1SM(IMTOFF),LOB,LOB,LOB,LOB)            
        END IF
*. Pack and diagonalize
        CALL TRIPAK(RHO1SM(IMTOFF),SCR,1,LOB,LOB)
        ONEM = -1.0D0
*. scale with -1 to get highest occupation numbers as first eigenvectors
        CALL SCALVE(SCR,ONEM,LOB*(LOB+1)/2)       
        CALL EIGEN(SCR,XNAT(IMTOFF),LOB,0,1)
*
        DO  I = 1, LOB   
          OCCNUM(IOBOFF-1+I) = - SCR(I*(I+1)/2) 
        END DO 
*. Order the degenerate eigenvalues so diagonal terms are maximized
        TESTY = 1.0D-11
        DO IOB = 2, LOB
          IF(ABS(OCCNUM(IOBOFF-1+IOB)-OCCNUM(IOBOFF-2+IOB))
     &       .LE.TESTY) THEN
            XII   = ABS(XNAT(IMTOFF-1+(IOB-1)  *LOB+IOB  ))
            XI1I1 = ABS(XNAT(IMTOFF-1+(IOB-1-1)*LOB+IOB-1))
            XII1  = ABS(XNAT(IMTOFF-1+(IOB-1-1)*LOB+IOB  ))
            XI1I  = ABS(XNAT(IMTOFF-1+(IOB-1)  *LOB+IOB-1))
*
            IF( XI1I.GT.XII.AND.XII1.GT.XI1I1 ) THEN
*. interchange orbital IOB and IOB -1
              CALL SWAPVE(XNAT(IMTOFF+(IOB-1)*LOB),
     &                    XNAT(IMTOFF+(IOB-1-1)*LOB),LOB)
              SS = OCCNUM(IOBOFF-1+IOB-1)
              OCCNUM(IOBOFF-1+IOB-1) = OCCNUM(IOBOFF-1+IOB)
              OCCNUM(IOBOFF-1+IOB)   = SS             
              write(6,*) ' Orbitals interchanged ',
     &        IOBOFF-1+IOB,IOBOFF-2+IOB
            END IF
          END IF
        END DO
*
        IF(NTEST.GE.1) THEN
          WRITE(6,*)
          WRITE(6,*) 
     &    ' Natural occupation numbers for symmetry = ', ISMOB
          WRITE(6,*)
     &    ' ==================================================='
          WRITE(6,*)
          CALL WRTMAT(OCCNUM(IOBOFF),1,LOB,1,LOB)
          IF(NTEST.GE.2 ) THEN
            WRITE(6,*)
            WRITE(6,*) ' Corresponding Eigenvectors '
            WRITE(6,*)
            CALL WRTMAT(XNAT(IMTOFF),LOB,LOB,LOB,LOB)
          END IF
        END IF
      END DO
*. ( End of loop over orbital symmetries )
*
      RETURN
      END 
*
      SUBROUTINE TRPAD(MAT,FACTOR,NDIM)
C
C  MAT(I,J) = MAT(I,J) + FACTOR*MAT(J,I)
C
      IMPLICIT REAL*8           (A-H,O-Z)
      REAL*8            MAT(NDIM,NDIM)
C
C
      DO 100 J = 1, NDIM
        DO 90 I = J, NDIM
          MAT(I,J) =MAT(I,J) + FACTOR * MAT(J,I)
  90    CONTINUE
 100  CONTINUE
C
C
      IF( ABS(FACTOR) .NE. 1.0D0 ) THEN
        FAC2 = 1.0D0 - FACTOR**2
        DO 200 I = 1, NDIM
         DO 190 J = 1, I - 1
           MAT(J,I) = FACTOR*MAT(I,J ) + FAC2 * MAT(J,I)
 190     CONTINUE
 200    CONTINUE
      ELSE
        IF(FACTOR .EQ. 1.0D0) THEN
        DO 300 I = 1, NDIM
         DO 290 J = 1, I - 1
            MAT(J,I) = MAT(I,J )
 290     CONTINUE
 300    CONTINUE
      ELSE
        DO 400 I = 1, NDIM
         DO 390 J = 1, I - 1
            MAT(J,I) =-MAT(I,J )
 390     CONTINUE
 400    CONTINUE
      END IF
      END IF
      RETURN
      END
*CADDB
      SUBROUTINE LCNHCN(LSCR)
*
* Amount of scratch Needed in the CNHCNM routine 
*
* Jeppe Olsen, September 1993
*
* Amount of Memory required: 2*NACTEL + MXCSFC**2 +
*                             6*MXDTFC+MXDTFC**2+MXCSFC*MXDTFC+
*                             MAX(MXDTFC*NACTEL+2*NACTEL,4*NACOB+2*NACTEL)
*
* Where NACTEL: Number of active electrons
*       NACOB : Number of active orbitals
*       MXCSFC: Max number of CSF's for given COnfiguration
*       MXDTFC: Max number of Combs for given configuration
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
*./SPINFO/, old
      COMMON/SPINFO_OLD/MULTSP,MS2P,
     &              MINOP,MAXOP,NTYP,NDPCNT(MXPCTP),NCPCNT(MXPCTP),
     &              NCNATS(MXPCTP,MXPCSM),NDTASM(MXPCSM),NCSASM(MXPCSM),
     &              NCNASM(MXPCSM)
*. NACTEL is obtained from lucinp
      INCLUDE 'lucinp.inc'
*. NACOB is obtained from orbinp
       INCLUDE 'orbinp.inc'
*./SPINFO/
*. MXCSFC, MXSDFC
      MXCSFC = 0
      MXDTFC = 0
      DO 100 ITYP = 1, NTYP
        MXCSFC = MAX(MXCSFC,NCPCNT(ITYP))
        MXDTFC = MAX(MXDTFC,NDPCNT(ITYP))
  100 CONTINUE
*
*
      LSCR  = 2*NACTEL + MXCSFC**2 +
     &        6*MXDTFC+MXDTFC**2+MXCSFC*MXDTFC+
     &        MAX(MXDTFC*NACTEL+2*NACTEL,4*NACOB+2*NACTEL)
*
C?    WRITE(6,*) ' LCNHCN: MXCSFC MXDTFC ',MXCSFC,MXDTFC
C?    WRITE(6,*) ' LCNHCN: LSCR ', LSCR                    
*
      RETURN
      END 
CADDE
    

      SUBROUTINE GATVCD(LU,LBLK,NGAT,IGAT,XGAT,SEGMNT,IPRT)
*
* Gather elements from a file LU
*
* XGAT(I) = Vector(IGAT(I))
*
* Jeppe Olsen, September 1993
*
      IMPLICIT REAL*8           (A-H,O-Z)
*. Input
      INTEGER IGAT(NGAT)
*. Output
      DIMENSION XGAT(NGAT)
*. Scratch
      DIMENSION SEGMNT(*)
*
      CALL REWINE(LU,-1)
*
      IBASE = 1
      IBLOCK = 0
*
*. Loop over blocks of file
*
 1000 CONTINUE
        IBLOCK = IBLOCK + 1
        CALL NEXREC(LU,LBLK,SEGMNT,IEND,LENGTH)
        IF(IPRT.GE.10)
     &  WRITE(6,*) LENGTH, ' elements in block ',IBLOCK
        IF(IEND.EQ.0) THEN
          IFIRST = IBASE
          ILAST = IBASE + LENGTH - 1
          DO 100 I = 1, NGAT
            IF(IFIRST .LE. IGAT(I) .AND. IGAT(I) .LE. ILAST ) 
     &      XGAT(I) = SEGMNT(IGAT(I)-IFIRST+1) 
C?          IF(IFIRST .LE. IGAT(I) .AND. IGAT(I) .LE. ILAST ) 
C?   &      write(6,*) ' Catch I IGAT(I) XGAT(I) ',
C?   &                         I,IGAT(I),XGAT(I)
  100     CONTINUE
          IBASE = IBASE + LENGTH
      IF(LBLK.LT.0) GOTO 1000
        END IF
*
      NTEST = 0
      NTEST = MAX(IPRT,NTEST)
      IF(NTEST.GE.5) THEN
       WRITE(6,*) ' Gathered vector from GATVCD '      
       CALL WRTMAT(XGAT,1,NGAT,1,NGAT)
      END IF
*
      RETURN
      END
      SUBROUTINE GATVCS(VECO,VECI,INDEX,NDIM)
* Gather vector alllwing for sign change
*
* VECO(I) = VECI(INDEX(I))
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION VECI(1),VECO(1),INDEX(1)
*
      DO 100 I = 1, NDIM
  100 VECO(I) = VECI(ABS(INDEX(I)))*SIGN(1,INDEX(I))
*
      RETURN
      END
      SUBROUTINE SCAVCS(VECO,VECI,INDEX,NDIM)
*
* Scatter vector with sign change
*
* vecO(abs(index(i))) = veci(i)*sign(index(i))
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION VECI(1),VECO(1),INDEX(1)
C
      DO 100 I = 1, NDIM
  100 VECO(ABS(INDEX(I))) = VECI(I)*SIGN(1,INDEX(I))
C
      RETURN
      END
      SUBROUTINE AJGAT(NIORB,IORB,CIN,COUT,IBOT,ITOP,KBOT,KTOP,
     &                 ICGRP,ICSM,ICTP,NCROW,I1,XI1S,NKBTC,KEND)
*
* obtain C(j,I,K) = +/-sum(J) <J!a+j!K>C(J,I)
* Kstrings in the range KMIN to KTOP are active and
* I strings int the range IBOT to ITOP
*
* j belongs to the orbitals given in IORB
*
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      INTEGER IORB(NIORB)
      DIMENSION CIN(NCROW,*)
*. Output
      DIMENSION COUT(*)
*. Scratch
      DIMENSION I1(*),XI1S(*)
*
      NIBTC = ITOP - IBOT + 1
      DO 100 JJORB = 1, NIORB
        JORB = IORB(JJORB)
*. mapping <J!a+jorb!K> for K
        MAXNK = KTOP-KBOT+1
        CALL ADST_GAS(JORB,1,ICTP,ICSM,ICGRP,KBOT,KTOP,
     &            I1,XI1S,MAXNK,NKBTC,KEND)
*.Gather  C Block
*. First index: JORB, second index: JaKb
        ICGOFF = 1 + (JJORB-1)*NKBTC*NIBTC
C           MATCG(CIN,COUT,NROWI,NROWO,NROW1I, NGCOL,IGAT,GATSGN )
        CALL MATCG(CIN,COUT(ICGOFF),NCROW,NIBTC,IBOT,
     &             NKBTC,I1,XI1S)
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' AJGAT, Gathered C block '
        WRITE(6,*) ' *********************** '
        DO 200 JJORB = 1, NIORB
          WRITE(6,*) ' Block for JORB = ' , IORB(JJORB)
          IOFF = 1 + (JJORB-1)*NIBTC*NKBTC
          CALL WRTMAT(COUT(IOFF),NIBTC,NKBTC,NIBTC,NKBTC)
  200   CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE ALLO_ALLO
*
* Dimensions and
* Allocation of static memory
*
* =====
* Input
* =====
*
* KFREE: Pointer to first element of free space
* Information in /LUCINP/,/ORBINP/,/CSYM/
*
* ======
* Output
* ======
* KFREE: First array of free space after allocation of
*         static memory
* /GLBBAS/,/CDIM/
*
*
* =======
* Version
* =======
*
* Modified Jan 1997                              
*           Fall 97 (KPGINT1 added )
*           Spring 99
*           and 2012
*
*. Last revision; July 2013; Jeppe Olsen; Subspace allocations changed

*. Input
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'csmprd.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'spinfo.inc'
      INCLUDE 'csfbas.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'gasstr.inc'
*.Output
      INCLUDE 'glbbas.inc'
      
*.1: One electron integrals( Complete matrix allocated )
      CALL MEMMAN(KINT1,NTOOB ** 2,'ADDS  ',2,'INT1  ')
*. A copy of the original UNMODIFIED 1-elecs ints
      CALL MEMMAN(KINT1O,NTOOB ** 2,'ADDS  ',2,'INT1O ')
*. Zero to avoid problems with elements that will not 
*. be initialized
      ZERO = 0.0D0
      CALL SETVEC(WORK(KINT1),ZERO,NTOOB**2)
*. Raw 1-electron integrals in input MO basis
      CALL  MEMMAN(KH,NTOOB**2,'ADDS  ',2,'H1    ')
*. 1-electron integrals with contribution  from two-electron terms from
*  explicitly declared inactive orbitals
      CALL MEMMAN(KHINA,NTOOB**2,'ADDS ',2,'H1INA ')
*. Overlap matrix in AO basis
      CALL MEMMAN(KSAO,NTOOB**2,'ADDS ',2,'SAO   ')
*
      CALL SETVEC(WORK(KINT1O),ZERO,NTOOB**2)
*.1.1: Inactive fock matrix
      CALL MEMMAN(KFI  ,NTOOB ** 2,'ADDS  ',2,'FI    ')
      CALL MEMMAN(KFIO ,NTOOB ** 2,'ADDS  ',2,'FIO   ')
*.1.2 Inactive Fock matrx in zero order space 
      CALL MEMMAN(KFIZ,NTOOB **2, 'ADDS  ',2,'FIZ    ')
*.1.3 Inactive Fock matrix for alpha and beta - spin
      CALL MEMMAN(KFI_AL,NTOOB**2, 'ADDS  ',2,'FI_AL ')
      CALL MEMMAN(KFI_BE,NTOOB**2, 'ADDS  ',2,'FI_BE ')
*. Inactive + active Fock matrix
      CALL MEMMAN(KFIFA,NTOOB**2, 'ADDS  ',2,'FIFA  ')
*. Active Fock matrix
      CALL MEMMAN(KFA,NTOOB**2,'ADDS  ',2,'FA    ')
*. Fock matrix
      CALL MEMMAN(KF,NTOOB**2,'ADDS  ',2,'FOCK  ')
*
      IF(I_DO_NORTCI.EQ.1) THEN
*. Also space for the extra Fock-matrix for the biorthonormal expansion
        CALL MEMMAN(KF2,NTOOB**2,'ADDS  ',2,'FOCK2 ')
      END IF


*. Malmqvist transformation matrix
      CALL MEMMAN(KTPAM,NTOOB**2,'ADDS  ',2,'TPAM  ')
*.2: Two electron integrals
      IF(NOINT.EQ.0.AND.INCORE.EQ.1.AND.ISVMEM.EQ.0) THEN
         CALL MEMMAN(KINT_2EMO,NINT2,'ADDS  ',2,'INT2  ')
*. For initial set of integrals
         CALL MEMMAN(KINT_2EINI,NINT2,'ADDS  ',2,'INT2_I')
      END IF
*. Pointers to symmetry block of integrals
      CALL MEMMAN(KPINT1,NBINT1,'ADDS  ',2,'PINT1 ')
      CALL MEMMAN(KPINT2,NBINT2,'ADDS  ',2,'PINT2 ')
*. Pointers to nonsymmetric one-electron integrals
      DO ISM = 1, NSMOB
*. triangular packed
        CALL MEMMAN(KPGINT1(ISM),NSMOB,'ADDS  ',2,'PGINT1')
*. no packing
        CALL MEMMAN(KPGINT1A(ISM),NSMOB,'ADDS  ',2,'PGIN1A')
      END DO
*. Symmetry of last index as a function of initial index
      CALL MEMMAN(KLSM1,NBINT1,'ADDS  ',2,'LSM1   ')
      CALL MEMMAN(KLSM2,NBINT2,'ADDS  ',2,'LSM2   ')
*.3 One-body density   
      CALL MEMMAN(KRHO1,NACOB ** 2,'ADDS  ',2,'RHO1  ')
*.3.1: One-body spin density
      IF(ISPNDEN.GE.1) THEN
        CALL MEMMAN(KSRHO1,NACOB **2, 'ADDS  ',2,'SRHO1 ')
      ELSE 
        KSRHO1 = 1
      END IF
      IF(ISPNDEN.GE.2) THEN
*. Two-body spin-density matrices
        LENSS = (NACOB*(NACOB+1)/2) ** 2
        CALL MEMMAN(KRHO2AA,LENSS,'ADDS  ',2,'RHO2AA')
        LENAB = NACOB**4
        CALL MEMMAN(KRHO2AB,LENAB,'ADDS  ',2,'RHO2AB')
        CALL MEMMAN(KRHO2BB,LENSS,'ADDS  ',2,'RHO2AA')
      ELSE
        KRHO2AA = 1
        KROH2AB = 1
        KRHO2BB = 1
      END IF
*.4  Two-body density matrix
      LRHO2 = NACOB**2*(NACOB**2+1)/2
      IF(IDENSI.EQ.2.AND.ISVMEM.EQ.0) 
     &CALL MEMMAN(KRHO2,LRHO2     ,'ADDS  ',2,'RHO2  ')
C     IF(IDENSI.GE.1.OR.ISPNDEN.GE.1) THEN
C       WRITE(6,*) 
C    &  'Space for density matrices over all orbital-spaces allocated'
C      END IF
*. Array for giving the orbital spaces in which density should be calculated
       CALL MEMMAN(KDENSSPC,NGAS,'ADDL  ',1,'DENSPC')
*. Arrays for going between complete ST ordered orbitals and ST ordering 
*. of the orbitals in the densities
       CALL MEMMAN(KDTFREORD,NTOOB,'ADDL  ',2,'DTFREO')
       CALL MEMMAN(KFTDREORD,NTOOB,'ADDL  ',2,'FTDREO')
*.4. Integrals (ij!kk) -(ik!kj), pointer and space
CJO   CALL MEMMAN(KPNIJ,NTOOB ** 2,'ADDS  ',1,'KPNIJ ')
CJO   CALL MEMMAN(KIJKK,NTOOB**2 *(NTOOB+1) / 2, 'ADDS  '
CJO  &            ,2,'KIJKK ')
      KPNIJ = 1
      KIJKK = 1
*
*. Allocate memory for explicit hamiltonian and roots
*
      IF(ISBSPC_SEL.NE.0) THEN
C     IF(MXP1+MXP2+MXQ. NE. 0 .OR. IPROCC.NE.0 ) THEN
C       NSBDET = MXP1 + MXP2 + MXQ
        NSBDET = MXP1
        NSBDETP = MAX(NSBDET,IPROCC)
        MXP = MXP1 + MXP2
*. Space for complete diagonalization
        CALL MEMMAN(KSBEVC,NSBDET**2,'ADDS  ',2,'KSBEVC')
        CALL MEMMAN(KSBEVL,NSBDET,'ADDS  ',2,'KSBEVL')
*. KSBIDT must be able to hold list of subspace dets / print dets
        LSCR = NSBDETP+1
        CALL MEMMAN(KSBIDT,LSCR,'ADDS  ',1,'KSBIDT')
*. ( NSBDET + 1 elements allocated since NSBDET+1 values are 
* obtained in order to check for degenerencies )
        CALL MEMMAN(KSBCNF,LSCR        ,'ADDS  ',1,'KSBCNF')
        CALL MEMMAN(KSBIA ,LSCR        ,'ADDS  ',1,'KSBIA ')
        CALL MEMMAN(KSBIB ,LSCR        ,'ADDS  ',1,'KSBIB ')
*. Note: KH0 is dimensioned so, that it may hold matrix, eigvalues and 
*. eigenvectors
        LH0 = MXP*(MXP+1)/2 + MXP1*MXQ + MXQ + NSBDETP+1 + 
     &        NSBDET + NSBDET**2
        CALL MEMMAN(KH0,LH0,'ADDS  ',2,'KH0   ')
*. Scratch space for manipulating with H0
        LH0SCR = 
     &  MAX(6*NSBDETP,4*NSBDETP+4*NOCOB,
     &      MXP1*(MXP1+1)/2+MXP1**2)
        CALL MEMMAN(KH0SCR,LH0SCR      ,'ADDS  ',2,'KH0SCR')
*. For subspace configurations
        IF(NOCSF.EQ.0) THEN
          CALL MEMMAN(KSBCNFOCC,LOCC_SUB,'ADDL  ',1,'SBCNOC')
          CALL MEMMAN(KSBCNFOP,NCONF_SUB,'ADDL  ',1,'SBCNOP')
        ELSE
         KSBCNFOCC = 0
         KSBCNFOP = 0
        END IF
      ELSE
        KSBEVC = 0 
        KSBEVL = 0
        KSBIDT = 0
        KSBCNF = 0
        KH0    = 0
        KH0SCR = 0
        KSBCNFOCC = 0
        KSBCNFOP = 0
      END IF ! ISBSPC_SEL .ne. 0
*. Space for LZ and L2  for all roots of a CI
        MXLROOT = MAX(NROOT,INI_NROOT)
        CALL MEMMAN(KLZEXP,MXLROOT,'ADDL  ',2,'LZEXP ')
        CALL MEMMAN(KL2EXP,MXLROOT,'ADDL  ',2,'L2EXP ')
*
*. indeces for pair of orbitals symmetry ordered
*. Lower half packed
      CALL MEMMAN(KINH1,NTOOB*NTOOB,'ADDS  ',2,'KINH1  ')
*. Complete form
      CALL MEMMAN(KINH1_NOCCSYM,NTOOB*NTOOB,'ADDS  ',2,'KINH1  ')
*
*. Length of MO-MO and MO_AO expansion file
      LMOMO = 0
      LMOAO = 0
      DO ISM = 1, NSMOB
        LMOMO = LMOMO + NMOS_ENV(ISM)*NMOS_ENV(ISM)
        LMOAO = LMOAO + NMOS_ENV(ISM)*NAOS_ENV(ISM)
      END DO
C     WRITE(6,*) ' LMOMO LMOAO ', LMOMO,LMOAO
      CALL MEMMAN(KMOMO  ,LMOMO ,'ADDS  ',2,'MOMO  ')
      CALL MEMMAN(KMOREF ,LMOMO ,'ADDS  ',2,'MOREF ')
*
      CALL MEMMAN(KMOAOIN,LMOAO ,'ADDS  ',2,'MOAOIN')
      CALL MEMMAN(KMOAOUT,LMOAO ,'ADDS  ',2,'MOAOUT')
      CALL MEMMAN(KMOAO_ACT,LMOAO ,'ADDS  ',2,'MOAOUT')
*
*. Space for bioorthonormal orbitals in terms of AO's
        CALL MEMMAN(KCBIO,LMOAO ,'ADDS  ',2,'CBIO  ')
*. Space for bioorthonormal orbitals in terms of the current MO's
        CALL MEMMAN(KCBIO2,LMOAO ,'ADDS  ',2,'CBIO2 ')


*. And the MO
*. Space for handling similarity transformed Hamiltonian- 
*. if required
      IF(ISIMTRH.EQ.1) THEN
*. SIMTRH .NE. 0 => Some kind of CC, 
*. Check whether it is a closed sholl or openshell case 
        CALL CC_AC_SPACES(1,IREFTYP)
        IF(IREFTYP.NE.2) THEN        
*. Set up a single set of integrals, assuming that 
*. alpha and beta parts of T1 are identical
         CALL MEMMAN(KINT1_SIMTRH,NTOOB**2, 'ADDL  ',2,'SIMTR1')
         CALL MEMMAN(KPINT1_SIMTRH,NSMOB,   'ADDL  ',2,'PSMTR1')
         IF(ISVMEM.EQ.0) THEN 
           LEN =  NINT2_NO_CCSYM
           CALL MEMMAN(KINT2_SIMTRH,LEN,      'ADDL  ',2,'SIMTR2')
           CALL MEMMAN(KPINT2_SIMTRH,NSMOB**3,'ADDL  ',2,'PSMTR2')
         ELSE
           KINT2_SIMTRH=1
           KPINT2_SIMTRH=1
         END IF
         KINT1_SIMTRH_A = 1
         KINT1_SIMTRH_B = 1
         KINT2_SIMTRH_AA = 1
         KINT2_SIMTRH_BB = 1
         KINT2_SIMTRH_AB = 1
         KPINT2_SIMTRH_AB = 1
        ELSE 
*. High spin single determinant is reference, T1 has 
*. different alpha and beta components.
         CALL MEMMAN(KINT1_SIMTRH_A,NTOOB**2, 'ADDL  ',2,'SMTHA ')
         CALL MEMMAN(KINT1_SIMTRH_B,NTOOB**2, 'ADDL  ',2,'SMTHB ')
         CALL MEMMAN(KPINT1_SIMTRH,NSMOB,   'ADDL  ',2,'PSMTR1')
*
         IF (ISVMEM.EQ.0) THEN
           LEN =  NINT2_NO_CCSYM
           CALL MEMMAN(KINT2_SIMTRH_AA,LEN,      'ADDL  ',2,'SMTHAA')
           CALL MEMMAN(KINT2_SIMTRH_BB,LEN,      'ADDL  ',2,'SMTHBB')
           LEN =  NINT2_NO_CCSYM_NO12SYM
           CALL MEMMAN(KINT2_SIMTRH_AB,LEN,      'ADDL  ',2,'SMTHAB')
           CALL MEMMAN(KPINT2_SIMTRH,NSMOB**3,'ADDL  ',2,'PSMTXX')
           CALL MEMMAN(KPINT2_SIMTRH_AB,NSMOB**3,'ADDL  ',2,'PSMTAB')
         ELSE
           KINT2_SIMTRH_AA = 1
           KINT2_SIMTRH_BB = 1
           KINT2_SIMTRH_AB = 1
           
         END IF
*
         KINT1_SIMTRH = 1
         KINT2_SIMTRH = 1
        END IF
      ELSE 
        KINT1_SIMTRH = 1
        KINT2_SIMTRH = 1
        KPINT1_SIMTRH = 1
        KPINT2_SIMTRH = 1
        KINT1_SIMTRH_A = 1
        KINT1_SIMTRH_B = 1
        KINT2_SIMTRH_AA = 1
        KINT2_SIMTRH_BB = 1
        KINT2_SIMTRH_AB = 1
        KPINT2_SIMTRH_AB = 1
      END IF
*. Supersymmetry info
      IF(I_USE_SUPSYM.EQ.1) THEN
*. Character labels
        CALL MEMMAN(KCSUPSYM_FOR_ORB,4*NTOOB,'ADDL  ',1,'CSUPSM')
*. L values
        CALL MEMMAN(KLVAL_FOR_ORB ,NTOOB,'ADDL  ',1,'L_OB  ')
        CALL MEMMAN(KMLVAL_FOR_ORB,NTOOB,'ADDL  ',1,'ML_OB ')
        CALL MEMMAN(KPA_FOR_ORB,NTOOB,'ADDL  ',1,'PA_OB ')
*. Tables for going between irrep and symmetries
*. Max number of irreps is based on ATOMIC or LINEAR supersymmetry
        NSUPSYM_MAX = MXPL + 1 + MXPL*(MXPL+1)
        NIRREP_MAX  = 2*MXPL + 1
        CALL MEMMAN(KL_FOR_SUPSYM,NSUPSYM_MAX,'ADDL  ',1,'L_SPSM')
        CALL MEMMAN(KML_FOR_SUPSYM,NSUPSYM_MAX,'ADDL  ',1,'M_SPSM')
        CALL MEMMAN(KPA_FOR_SUPSYM,NSUPSYM_MAX,'ADDL  ',1,'P_SPSM')
        CALL MEMMAN(KIRREP_FOR_SUPSYM,NSUPSYM_MAX,'ADDL  ',1,'I_SPSM')
        CALL MEMMAN(KNSUPSYM_FOR_IRREP,NIRREP_MAX,'ADDL  ',1,'N_SPSM')
        CALL MEMMAN(KIBSUPSYM_FOR_IRREP,NIRREP_MAX,'ADDL  ',1,'B_SPSM')
        CALL MEMMAN(KISUPSYM_FOR_IRREP,NIRREP_MAX,'ADDL  ',1,'ISM_IR')
*. Supersymmetry for basis function
        CALL MEMMAN(KISUPSYM_FOR_BAS,NTOOB,'ADDL  ',1,'ISP_BS')
*. Info on orbitals with given super and standard symmetry
        CALL MEMMAN(KNBAS_FOR_SUP_STA_SYM,NSUPSYM_MAX*NSMOB,'ADDL  ',1,
     &       'NB_SSS')
        CALL MEMMAN(KIBBAS_FOR_SUP_STA_SYM,NSUPSYM_MAX*NSMOB,'ADDL  ',1,
     &       'BB_SSS')
        CALL MEMMAN(KIBAS_FOR_SUP_STA_SYM, NTOOB,'ADDL  ',1,'IB_SSS')
*. Info for going between irreps and symmetry-ordered orbitals
        CALL MEMMAN(KISHELL_FOR_BAS,NTOOB,'ADDL  ',1,'IR_BAS')
        CALL MEMMAN(KNBAS_FOR_SHELL,NTOOB,'ADDL  ',1,'NB_IRR')
        CALL MEMMAN(KIBBAS_FOR_SHELL,NTOOB,'ADDL  ',1,'IB_IRR')
        CALL MEMMAN(KIBAS_FOR_SHELL,NTOOB,'ADDL  ',1,'I_IRR ')
      ELSE
        KCSUPSYM_FOR_ORB = -1
        KLVAL_FOR_ORB = -1
        KMLVAL_FOR_ORB = -1
        KL_FOR_SUPSYM = -1
        KML_FOR_SUPSYM = -1
        KIRREP_FOR_SUPSYM = -1
        KNSUPSYM_FOR_IRREP = -1
        KIBSUPSYM_FOR_IRREP = -1
        KISUPSYM_FOR_IRREP = -1
*
        KISUPSYM_FOR_BAS = -1
        KNBAS_FOR_SUP_STA_SYM = -1
        KIBBAS_FOR_SUP_STA_SYM = -1
        KIBAS_FOR_SUP_STA_SYM = -1
*
        KISHELL_FOR_BAS = -1
        KNBAS_FOR_SHELL = -1
        KIBBAS_FOR_SHELL = -1
        KIBAS_FOR_SHELL = -1
*
      END IF
      CALL MEMMAN(KMO_STA_TO_ACT_REO,NTOOB,'ADDL  ',1,'MOSTSP') 
      CALL MEMMAN(KMO_SUPSYM,NTOOB,'ADDL  ',1,'MOSPSM') 
      CALL MEMMAN(KMO_GNSYM ,NTOOB,'ADDL  ',1,'MOGNSM') 
      CALL MEMMAN(KIREO_GNSYM_TO_TS_ACOB,NTOOB,'ADDL  ',1,
     &            'REOGAC')
      CALL MEMMAN(KMO_STA_SUPSYM,NTOOB,'ADDL  ',1,'STSPSM')
      CALL MEMMAN(KMO_OCC_SUPSYM,NTOOB,'ADDL  ',1,'OCSPSM')
      CALL MEMMAN(KMO_ACT_SUPSYM,NTOOB,'ADDL  ',1,'ACSPSM')
      CALL MEMMAN(KIREO_INI_OCC, NTOOB,'ADDL  ',1,'INOCRE')
*. Some arrays allowing hiding the use of super-symmetry
      IF(I_USE_SUPSYM.EQ.0) THEN
        NGENSMOB = NSMOB
      ELSE
COLD    NGENSMOB = NSMOB*NSUPSYM_MAX
        NGENSMOB = NSUPSYM_MAX
      END IF
      
*
* Allocation of memory for configurations, Reorder vectors...
* 
      IF(NOCSF.EQ.0) THEN
*. Memory for prototype info on configs and CSF's
*. And allocate memory for prototype info
       CALL MEMMAN(KDFTP,LPDT_OCC,'ADDL  ',1,'DFTP  ')
       CALL MEMMAN(KCFTP,LPCS_OCC,'ADDL  ',1,'CFTP  ')
       CALL MEMMAN(KDTOC,LPDTOC,'ADDL  ',2,'D_TO_C')
* Arrays for addressing prototype determinants for each prototype-config
       DO IOPEN = MINOP, MAXOP
         ITYP = IOPEN + 1
*
         IALPHA = (IOPEN+MS2)/2
         LZ = IOPEN*IALPHA        
         LPTDT = IBION(IOPEN,IALPHA)
         CALL MEMMAN(KZ_PTDT(ITYP),LZ,'ADDL  ',1,'Z_PTDT')
         CALL MEMMAN(KREO_PTDT(ITYP),LPTDT,'ADDL  ',1,'RE_PTD')
       END DO
*
*. Info on vectors to be stored:
*
       IF(ICNFBAT.EQ.1) THEN
*. Complete vectors
         LLCONFOCC_MAX = LCONFOCC_MAX
         NNCONF_AS_MAX = NCONF_AS_MAX
         NNCM_MAX = NCM_MAX
       ELSE
*. Blocks of vectors
         LLCONFOCC_MAX = LEN_OCC_FOR_OCCLS_MAX
         NNCONF_AS_MAX = NCN_ALLSYM_FOR_OCCLS_MAX
         NNCM_MAX = NCM_FOR_OCCLS_MAX
       END IF
*
       CALL MEMMAN(KZCONF,NOCOB*NACTEL*2,'ADDL  ',1,'ZCONF ')
       CALL MEMMAN(KICONF_OCC(IREFSM),LLCONFOCC_MAX,'ADDL  ',1,'CNFOCC')
*. Reorder array for configurations
       CALL MEMMAN(KICONF_REO(1),NNCONF_AS_MAX,'ADDL  ',1,'CNFREO')
C* Symmetry for reorder of determinants for reference symmetry
       CALL MEMMAN(KSDREO_I(IREFSM),NNCM_MAX,'ADDL  ',1,'SDREOI')
*. _S not active:
       KSDREO_S(IREFSM) = 1
       IF(I_DO_SBCNF.EQ.1) THEN
*. Allocate space for occupation of subconfigurations
C IFRMR(WORK,IROFF,IELMNT)
         DO JOCSBCLS = 1, NOCSBCLST
C?         WRITE(6,*) ' JOCSBCLS = ', JOCSBCLS
           LOCC = IFRMR(dbl_mb(KLSBCNF),1,JOCSBCLS)
C?         WRITE(6,*) ' LOCC = ', LOCC
           CALL MEMMAN(LPOINT,LOCC,'ADDL  ',1,'OCSBCN')
C?         WRITE(6,*) ' LPOINT = ', LPOINT
C ICOPVE2(IIN,IOFF,NDIM,IOUT)
C          CALL ICOPVE2(LPOINT,JOCSBCLS,1,WORK(KKSBCNF))
C               ICOPVE3(IIN,IOFFIN,IOUT,IOFFOUT,NDIM)
           CALL ICOPVE3(LPOINT,1,dbl_mb(KKOCSBCNF),JOCSBCLS,1)
         END DO
*
C?       WRITE(6,*) ' Offset to occ of subconfs: '
C?       CALL IWRTMA(WORK(KKOCSBCNF),1,NOCSBCLST,1,NOCSBCLST)
       END IF ! I_DO_SBCNF
      END IF !csf's are in use
*
*. Allocate storage of two blockings of CI-vectors
*
      IATP = 1
      IBTP = 2
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
C?    WRITE(6,*) ' TEST: NOCTPA, NOCTPB = ',  NOCTPA, NOCTPB
*
      IGENERAL = 1
      IF(IGENERAL.EQ.1) THEN
        LEN_IOIO= NSPGPFTP_MAX**2
      ELSE
        LEN_IOIO = NOCTPA*NOCTPB
      END IF
      CALL MEMMAN(KCIOIO,LEN_IOIO,'ADDL  ',1,'CIOIO ')
      CALL MEMMAN(KSIOIO,LEN_IOIO,'ADDL  ',1,'CSOIO ')
*
      CALL MEMMAN(KCBLTP,NSMST,'ADDL  ',1,'CBLTP ')
      CALL MEMMAN(KSBLTP,NSMST,'ADDL  ',1,'CSLTP ')
*
      NTTS = MXNTTS
C?    WRITE(6,*) ' ALLO: NTTS = ', NTTS
      CALL MEMMAN(KCLBT ,NTTS  ,'ADDL  ',1,'CLBT  ')
      CALL MEMMAN(KCLBLK,NTTS  ,'ADDL  ',1,'CLBLK ')
      CALL MEMMAN(KCLEBT,NTTS  ,'ADDL  ',1,'CLEBT ')
      CALL MEMMAN(KCI1BT,NTTS  ,'ADDL  ',1,'CI1BT ')
      CALL MEMMAN(KCIBT ,8*NTTS,'ADDL  ',1,'CIBT  ')
      CALL MEMMAN(KC2B  ,  NTTS,'ADDL  ',1,'C2BT  ')
*
      CALL MEMMAN(KSLBT ,NTTS  ,'ADDL  ',1,'SLBT  ')
      CALL MEMMAN(KSLBLK,NTTS  ,'ADDL  ',1,'SLBLK ')
      CALL MEMMAN(KSLEBT,NTTS  ,'ADDL  ',1,'SLEBT ')
      CALL MEMMAN(KSI1BT,NTTS  ,'ADDL  ',1,'SI1BT ')
      CALL MEMMAN(KSIBT ,8*NTTS,'ADDL  ',1,'SIBT  ')
      CALL MEMMAN(KS2B  ,  NTTS,'ADDL  ',1,'S2BT  ')
*. Info in batches of occupation blocks
      CALL MEMMAN(KCNOCCLS_BAT,NOCCLS_MAX,'ADDL  ',1,'NOCBAT')
      CALL MEMMAN(KCIBOCCLS_BAT,NOCCLS_MAX,'ADDL  ',1,'NOCBAT')
      CALL MEMMAN(KSNOCCLS_BAT,NOCCLS_MAX,'ADDL  ',1,'NOCBAT')
      CALL MEMMAN(KSIBOCCLS_BAT,NOCCLS_MAX,'ADDL  ',1,'NOCBAT')
*. (more to come )
      CALL MEMMAN(IDUMMY,IDUMMY,'CHECK',IDUMMY,'Dummy ')
      RETURN
      END
      SUBROUTINE ANACI4(NAEL,IASTR,NBEL,IBSTR,
     &                  CI,NSMST,
     &                  ISMOST,IBLTP,
     &                  NSSOA,NSSOB,IOCOC,NOCTPA,NOCTPB,
     &                  ISSOA,ISSOB,LUC,
     &                  ICISTR)
*
* Intitial analyzer: Print out all coefficients !!
* Turbo-ras version
*
* ========================
* General symmetry version
* ========================
*
* Jeppe Olsen, Winter of 1991
*
      IMPLICIT REAL*8           (A-H,O-Z)
*.General input
      DIMENSION NSSOA(NOCTPA,*),NSSOB(NOCTPB,* )
      DIMENSION ISSOA(NOCTPA,*),ISSOB(NOCTPB,*)
      DIMENSION IASTR(NAEL,*),IBSTR(NBEL,*)
*. Specific input
      DIMENSION IOCOC(NOCTPA,NOCTPB)
      DIMENSION ISMOST(*),IBLTP(*)
      DIMENSION CI(*)
*
      WRITE(6,*) ' List of All CI coefficients '
      WRITE(6,*) ' =========================== '
      WRITE(6,*)
      WRITE(6,*)
     & ' Form of output: alpha string beta string coefficient'
*
      IDET = 0
      DO 1000 IASM = 1, NSMST
        IBSM = ISMOST(IASM)
        IF(IBSM.EQ.0.OR.IBLTP(IASM).EQ.0) GOTO 1000
        IF(IBLTP(IASM).EQ.2) THEN
          IREST1 = 1
        ELSE
          IREST1 = 0
        END IF
*
        DO 999  IATP = 1,NOCTPA
          IF(IREST1.EQ.1) THEN
            MXBTP = IATP
          ELSE
            MXBTP = NOCTPB
          END IF
          DO 900 IBTP = 1,MXBTP
          IF(IOCOC(IATP,IBTP) .EQ. 0 ) GOTO 900
          IBSTRT = ISSOB(IBTP,IBSM)
          IBSTOP = IBSTRT + NSSOB(IBTP,IBSM)-1
          DO 899 IB = IBSTRT,IBSTOP
            IBREL = IB - IBSTRT + 1
            IF(IREST1.EQ.1.AND.IATP.EQ.IBTP) THEN
              IASTRT = ISSOA(IATP,IASM) - 1 + IBREL
            ELSE
              IASTRT = ISSOA(IATP,IASM)
            END IF
            IASTOP = ISSOA(IATP,IASM) + NSSOA(IATP,IASM) - 1
            DO 800 IA = IASTRT,IASTOP
              IDET = IDET + 1
              WRITE(6,'(4I3,3X,E20.13)') (IASTR(IAEL,IA),IAEL=1,NAEL),
     &                                (IBSTR(IBEL,IB),IBEL=1,NBEL),
     &                                 CI(IDET)
     &        
  800       CONTINUE
  899     CONTINUE
  900   CONTINUE
  999   CONTINUE
*
 1000 CONTINUE
*
      RETURN
      END
      SUBROUTINE ANACIS(C,LUC,NSSOA,NSSOB,NOCTPA,NOCTPB,
     &                 THRES,MAXTRM,ISSOA,ISSOB,NAEL,NBEL,IOCOC,
     &                 IASTR,IBSTR,ISMOS,IBLTP,NSMST,IUSLAB,
     &                 IOBLAB,NCPMT,WCPMT,MNRS1,MXRS1,MNRS3,MXRS3,
     &                 IEL1A,IEL3A,IEL1B,IEL3B,ICISTR)
*
* Analyze CI vector:
*
*      1) Print atmost MAXTRM  combinations with coefficients
*         larger than THRES
*
*      2) Number of coefficients in given range
*
*      3) Number of coefficients in given range for given 
*         occupation of RAS1,RAS3
*
* Jeppe Olsen , Jan. 1989 ,   
*     Revision May 1992 ( Labels added )
*              July 1993: Lucia adapted + printout for each Ras class
*                                                  

*
*. If IUSLAB  differs from zero Character*6 array IOBLAB is used to identify
*  Orbitals
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION C(*)
      DIMENSION NSSOA(NOCTPA,*),NSSOB(NOCTPB,*)
      DIMENSION ISSOA(NOCTPA,*),ISSOB(NOCTPB,*)
      DIMENSION IASTR(NAEL,*),IBSTR(NBEL,*)
      DIMENSION IOCOC(NOCTPA,NOCTPB)
      DIMENSION ISMOS(*),IBLTP(*)
      CHARACTER*6 IOBLAB(*)
      DIMENSION IEL1A(NOCTPA),IEL3A(NOCTPA)
      DIMENSION IEL1B(NOCTPB),IEL3B(NOCTPB)
*. Output
      DIMENSION NCPMT(10,(MXRS1-MNRS1+1),(MXRS3-MNRS3+1))
      DIMENSION WCPMT(10,(MXRS1-MNRS1+1),(MXRS3-MNRS3+1))
*
      IF(IUSLAB.NE.0) THEN 
       WRITE(6,*)
       WRITE(6,*) 
     & ' Labels for orbitals are of the type n l ml starting with n = 1'
       WRITE(6,*) 
     & ' so the user should not be  alarmed by labels like 1 f+3 '  
       WRITE(6,*)
      END IF
     
C     WRITE(6,*) 'C(1) = ',C(1)
      MINPRT = 0
      ITRM = 0
      IDET = 0
      IIDET = 0
      ILOOP = 0
      NCIVAR = 0
      IF(THRES .LT. 0.0D0 ) THRES = ABS(THRES)
      CNORM = 0.0D0
2001  CONTINUE
      IF( ICISTR .GE. 2 ) CALL REWINO(LUC)
      IIDET = 0
      ILOOP = ILOOP + 1
      IF ( ILOOP  .EQ. 1 ) THEN
        XMAX = 1.0D0
        XMIN = 1.0D0/SQRT(10.0D0)
      ELSE
        XMAX = XMIN
        XMIN = XMIN/SQRT(10.0D0)
      END IF
      IF(XMIN .LT. THRES  ) XMIN =  THRES
      IDET = 0
C
      WRITE(6,*)
      WRITE(6,*)
      WRITE(6,'(A,E10.4,A,E10.4)')
     &'  Printout of coefficients in interval  ',XMIN,' to ',XMAX
      WRITE(6,'(A)')
     &'  ========================================================='
      WRITE(6,*)
*
      DO 100 IASM = 1, NSMST
        IBSM = ISMOS(IASM)
C       write(6,*) ' Iasm ibsm ', iasm,ibsm
        IF(IBSM.EQ.0.OR.IBLTP(IASM).EQ.0) GOTO 100
*
        DO 95 IATP = 1, NOCTPA
C       write(6,*) ' iatp ', iatp
        IF(IBLTP(IASM).EQ.2) THEN
          IRESTR = 1
          MXBTP = IATP
        ELSE
          IRESTR = 0
          MXBTP = NOCTPB
        END IF
        DO 94 IBTP = 1, MXBTP
C       write(6,*) ' ibtp ', ibtp
        IF( IOCOC(IATP,IBTP) .LE. 0 ) GOTO 94
C        write(6,*) ' iococ test passed '
*
        IABAS = ISSOA(IATP,IASM)
        IBBAS = ISSOB(IBTP,IBSM)
*
        NIA = NSSOA(IATP,IASM)
        NIB = NSSOB(IBTP,IBSM)
*
        IF( ICISTR.GE.2 ) THEN 
*. Read in a Type-Type-symmetry block
          CALL IFRMDS(IDET,1,-1,LUC)
          CALL FRMDSC(C,IDET,-1,LUC,IMZERO,IAMPACK)
          IDET = 0
        END IF

        DO 90 IB = IBBAS,IBBAS+NIB-1
          IF(IRESTR.EQ.1.AND.IATP.EQ.IBTP) THEN
            MINIA = IB - IBBAS + IABAS
          ELSE
            MINIA = IABAS
          END IF
          DO 80 IA = MINIA,IABAS+NIA-1
*
            IF(ILOOP .EQ. 1 ) NCIVAR = NCIVAR + 1
            IDET = IDET + 1
C           WRITE(6,*) ' IASM IBSM IA IB IDET ',IASM,IBSM,IA,IB,IDET
            IF( XMAX .GE. ABS(C(IDET)) .AND.
     &      ABS(C(IDET)).GT. XMIN ) THEN
              ITRM = ITRM + 1
              IIDET = IIDET + 1
              IF( ITRM .LE. MAXTRM ) THEN
                CNORM = CNORM + C(IDET) ** 2
                WRITE(6,'(A)')
                WRITE(6,'(A)')
     &          '                 =================== '
                WRITE(6,*)
*
                WRITE(6,'(A,I8,A,E14.8)')
     &          '  Coefficient of combination ',IDET,' is ',
     &          C(IDET)
                WRITE(6,'(A)')
     &          '  Corresponding alpha - and beta string '
                IF(IUSLAB.EQ.0) THEN
                  WRITE(6,'(4X,10I4)')
     &            (IASTR(IEL,IA),IEL = 1, NAEL )
                  WRITE(6,'(4X,10I4)')
     &            (IBSTR(IEL,IB),IEL = 1, NBEL )
                ELSE 
                  WRITE(6,'(4X,10(1X,A6))')
     &            (IOBLAB(IASTR(IEL,IA)),IEL = 1, NAEL )
                  WRITE(6,'(4X,10(1X,A6))')
     &            (IOBLAB(IBSTR(IEL,IB)),IEL = 1, NBEL )
                END IF
              END IF
            END IF
   80     CONTINUE
   90   CONTINUE
   94   CONTINUE
   95  CONTINUE
  100 CONTINUE
       IF(IIDET .EQ. 0 ) WRITE(6,*) '   ( no coefficients )'
       IF( XMIN .GT. THRES .AND. ILOOP .LE. 30 ) GOTO 2001
*
       WRITE(6,'(A,E15.8)')
     & '  Norm of printed CI vector .. ', CNORM
*
*.Size of CI coefficients
*
*
      IDET = 0
      IF(ICISTR .GE. 2 ) CALL REWINO(LUC)
      CALL ISETVC(NCPMT,0    ,10*(MXRS1-MNRS1+1)*(MXRS3-MNRS3+1))
      CALL SETVEC(WCPMT,0.0D0,10*(MXRS1-MNRS1+1)*(MXRS3-MNRS3+1))
C     write(6,*) ' Jest before loop 200 '
      DO 200 IASM = 1, NSMST
        IBSM = ISMOS(IASM)
C       write(6,*) ' iasm ibsm ', iasm,ibsm
        IF(IBSM.EQ.0.OR.IBLTP(IASM).EQ.0) GOTO 200
*
        DO 195 IATP = 1, NOCTPA
        IF(IBLTP(IASM).EQ.2) THEN
          IRESTR = 1
          MXBTP = IATP
        ELSE
          IRESTR = 0
          MXBTP = NOCTPB
        END IF
        NEL1A = IEL1A(IATP)
        NEL3A = IEL3A(IATP)
        DO 194 IBTP = 1, MXBTP
C       write(6,*) ' iatp ibtp ', iatp, ibtp 
        IF( IOCOC(IATP,IBTP) .LE. 0 ) GOTO 194
*
        NEL1B = IEL1B(IBTP)
        NEL3B = IEL3B(IBTP)
*
        NEL1 = NEL1A + NEL1B
        NEL3 = NEL3A + NEL3B
*
        IABAS = ISSOA(IATP,IASM)
        IBBAS = ISSOB(IBTP,IBSM)
*
        NIA = NSSOA(IATP,IASM)
        NIB = NSSOB(IBTP,IBSM)
*
        IF( ICISTR.GE.2 ) THEN 
*. Read in a Type-Type-symmetry block
          CALL IFRMDS(IDET,1,-1,LUC)
          CALL FRMDSC(C,IDET,-1,LUC,IMZERO,IAMPACK)
          IDET = 0
        END IF

        DO 190 IB = IBBAS,IBBAS+NIB-1
          IF(IRESTR.EQ.1.AND.IATP.EQ.IBTP) THEN
            MINIA = IB - IBBAS + IABAS
          ELSE
            MINIA = IABAS
          END IF
          DO 180 IA = MINIA,IABAS+NIA-1
*
            IDET = IDET + 1
C           write(6,*) ' IDET C ', IDET,C(IDET)
            DO 170 IPOT = 1, 10
              IF(10.0D0 ** (-IPOT+1).GE.ABS(C(IDET)).AND.
     &           ABS(C(IDET)).GT. 10.0D0 ** ( - IPOT )) THEN
                 NCPMT(IPOT,NEL1-MNRS1+1,NEL3-MNRS3+1) = 
     &           NCPMT(IPOT,NEL1-MNRS1+1,NEL3-MNRS3+1) + 1  
*
                 WCPMT(IPOT,NEL1-MNRS1+1,NEL3-MNRS3+1) = 
     &           WCPMT(IPOT,NEL1-MNRS1+1,NEL3-MNRS3+1) + 
     &           C(IDET) ** 2
              END IF
  170       CONTINUE
              

            
  180     CONTINUE
  190   CONTINUE
  194   CONTINUE
  195  CONTINUE
  200 CONTINUE
*23456
      WRITE(6,'(A)')
      WRITE(6,'(A)') '   Magnitude of CI coefficients '
      WRITE(6,'(A)') '  =============================='
      WRITE(6,'(A)')
      WACC = 0.0D0
      NACC = 0
      DO 300 IPOT = 1, 10
        W = 0.0D0
        N = 0
        DO 290 IEL1 = MNRS1,MXRS1
          DO 280 IEL3 = MNRS3,MXRS3
            N = N + NCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
            W = W + WCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
C           write(6,*) ' IPOT IEL1 IEL3 N W '
C           write(6,*)  IPOT,IEL1,IEL3,N,W 
  280     CONTINUE
  290   CONTINUE
        WACC = WACC + W
        NACC = NACC + N
        WRITE(6,'(A,I2,A,I2,3X,I7,3X,E15.8,3X,E15.8)')
     &  '  10-',IPOT,' TO 10-',(IPOT-1),N,W,WACC           
  300 CONTINUE
*
      WRITE(6,*) ' Number of coefficients less than  10-11',
     &           ' IS  ',NCIVAR - NACC
*
      IF(MNRS1.NE.MXRS1.OR.MNRS3.NE.MXRS3) THEN
      WRITE(6,'(A)')
      WRITE(6,'(A)') 
     & '   Magnitude of CI coefficients for each excitation level '
      WRITE(6,'(A)') 
     & '  ========================================================='
      WRITE(6,'(A)')
      DO 400 IEL1 = MNRS1, MXRS1
        DO 390 IEL3 = MNRS3, MXRS3
          N = 0
          DO 380 IPOT = 1, 10
            N = N + NCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
  380     CONTINUE
          IF(N .NE. 0 ) THEN
            WRITE(6,*)
            WRITE(6,'(A,2I3)')
     &      '         Occupation of RAS 1 and RAS 3: ', IEL1, IEL3 
            WRITE(6,'(A,I9)')  
     &      '         Number of coefficients larger than 10-11 ', N
            WRITE(6,*)
            WACC = 0.0D0
            DO 370 IPOT = 1, 10
              N =  NCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
              W =  WCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
              WACC = WACC + W
              WRITE(6,'(A,I2,A,I2,3X,I7,3X,E15.8,3X,E15.8)')
     &        '  10-',IPOT,' TO 10-',(IPOT-1),N,W,WACC           
  370       CONTINUE
          END IF 
  390   CONTINUE
  400 CONTINUE
*
*. Total weight and number of dets per excitation level
*
      WRITE(6,'(A)')
      WRITE(6,'(A)') 
     & '   Total weight and number of SD''s (> 10 ** -11 ) : '
      WRITE(6,'(A)') 
     & '  ================================================='
      WRITE(6,'(A)')
      WRITE(6,*) ' Ras 1  Ras3        N      Weight      Acc. Weight '
      WRITE(6,*) ' ==================================================='
      WACC = 0.0D0
      DO 500 IEL3 = MNRS3, MXRS3
        DO 490 IEL1 = MXRS1, MNRS1,-1
          N = 0
          W = 0.0D0
          DO 480 IPOT = 1, 10
            N = N + NCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
            W = W + WCPMT(IPOT,IEL1-MNRS1+1,IEL3-MNRS3+1)
  480     CONTINUE
          WACC = WACC + W
          IF(N .NE. 0 ) THEN
            WRITE(6,'(1X,I4,2X,I4,3X,I8,4X,E8.3,7X,E8.3)') 
     &      IEL1,IEL3,N,W,WACC
          END IF
  490   CONTINUE
  500 CONTINUE
      END IF
*
      RETURN
      END
C    &     NBATCH,WORK(KCLBT),WORK(KCLEBT),WORK(KCLBLK),WORK(KCI1BT),
      SUBROUTINE CIEIG5(MV7,INICI,EROOT,VEC1,
     &           VEC2,MINST,LUDIA,LU1,LU2,LU3,LU4,LU5,LU6,LU7,LU8,
     &           NDIM,NBLK,NROOT,MAXVEC,MXCIIT,LUINCI,
     &           IPRT,PEIGVC,NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,
     &           EIGSHF,ICISTR,LBLK,IDIAG,VEC3,THRES_E,
     &           NBATCH,
     &           ICBLBT,ICLEBT,ICLBLK,ICI1BT,ICBLOCK,
     &           ISBLBT,ISLEBT,ISLBLK,ISI1BT,ISBLOCK,
     &           INIDEG,
     &           E_THRE,C_THRE,E_CONV,C_CONV,ICLSSEL,IBLK_TO_CLS,
     &           NCLS,CLS_C,CLS_E,CLS_CT,CLS_ET,CLS_A,ICLS_L,RCLS_L,
     &           BLKS_A,
     &           CLS_DEL,CLS_DELT,CLS_GAMMA,CLS_GAMMAT,ISKIPEI,I2BLK,
     &           ICLS_A2,MXLNG,
     &           IROOT_SEL,IBASSPC,EBASC,CBASC,NSPC,
     &           MULSPC,IPAT,LPAT,ISPC,NCNV_RT,IPRECOND,IUSEH0P,
     &           MPORENP_E,RNRM_CNV,CONVER,ISBSPPR_ACT,ILAST)
*
* Master routine for CI diagonalization
*
* Modified to handle PQ - preconditioner , May 1990
* PICO,MICDV4 added spring of 1991
*
*                    Nov. 7, 2012; Jeppe Olsen; ISBLOCK added for PICO
*                    Nov. 9, 2012; Jeppe Olsen; CLS_GAMMA, CLS_GAMMAT added
*                    Jan. 6, 2013; Jeppe Olsen; ISBSPPR_ACT added
* Last modification; Feb. 12,2013; Jeppe Olsen; IROOT_SEL replacing IROOTHOMING
*
      IMPLICIT REAL*8(A-H,O-Z)
      LOGICAL CONVER
      DIMENSION VEC1(*),VEC2(*)
C     DIMENSION INIDET(100)
      PARAMETER( LLWRK =180000 )
      COMMON/SCR/SCR1(LLWRK),ISCR1(LLWRK)
*. Output from Subspace dagonalization
      DIMENSION H0(*),IPNTR(*),H0SCR(*),PEIGVC(*)
      DIMENSION RNRM_CNV(*)
      DIMENSION ICBLOCK(*), ISBLOCK(*)
*. 
      DIMENSION EROOT(NROOT)
      IF( IPRT.GT.1) WRITE(6,*)
      IF( IPRT.GT. 1 )  WRITE(6,'(/A)')
     &'          *** information from ci diagonalization  ***'
C?    WRITE(6,*)
C?    WRITE(6,*) ' IROOT_SEL in CIEIG5 ', IROOT_SEL
C?    WRITE(6,*) ' INIDEG in CIEIG5 ', INIDEG
C?    WRITE(6,*) ' NCNV_RT = ', NCNV_RT
C?    WRITE(6,*) ' IPRECOND = ', IPRECOND
C?    WRITE(6,*) ' ISBSPPR_ACT = ', ISBSPPR_ACT
C?    WRITE(6,*) ' ICISTR,LBLK = ', ICISTR,LBLK
C?    WRITE(6,*) ' IPRT = ', IPRT
C?    WRITE(6,*) ' ILAST in CIEIG5 ', ILAST 
      NTEST=0
*
*               ====================================
** 1:               INITIAL VARIATIONAL SUBSPACE
*               ====================================
*
      IF( INICI .EQ. 0 ) THEN
        IF(NPRDET .EQ. 0 ) THEN
* ==================================================
*. Initial guess from lowest elements of CI diagonal
* ==================================================
* In order treat degenerencies, the lowest 4 * NROOT elements are
*.obtained
C?        write(6,*) ' CIEIG5 NDIM NROOT ', NDIM,NROOT
          NFINDM = MIN(NDIM,4*NROOT)
          CALL FNDMND(LUDIA,LBLK,VEC1,NFINDM,NFINDA,ISCR1(1+2*NFINDM),
     &                SCR1(1+2*NFINDM),ISCR1,SCR1,IPRT)
          CALL REWINO(LU1)
          IBASE = 1
          TEST = 1.0D-10
          IWPRNT = 0
          DO 100 IROOT = 1, NROOT
*. Number of degenerate elements
            NDEG = 1
            XVAL = SCR1(IBASE)
   90       CONTINUE
            IF(IBASE-1+NDEG+1.LE.NFINDA) THEN
              IF (ABS(SCR1(IBASE-1+NDEG+1)-XVAL).LE.TEST) THEN
                NDEG = NDEG + 1
                GOTO 90
              END IF
            END IF
C?          WRITE(6,*) ' IROOT NDEG ', IROOT,NDEG
*
            IF(INIDEG.EQ.0.AND.NDEG.GT.1) THEN
             IF(IWPRNT.EQ.0) THEN
              WRITE(6,*) ' WARNING WARNING WARNING WARNING ! '
              WRITE(6,*) ' DEGENERATE INITIAL VECTORS FOR CI '
              WRITE(6,*) ' I AM NOT ALLOWED TO TAKE THIS INTO '
              WRITE(6,*) ' CONSIDERATION SINCE  INIDEG = 0 '
              WRITE(6,*)
              WRITE(6,*) ' I hope you know what you are doing '
              IWPRNT = 1
             END IF
             NDEG = 1
            END IF

*. Initial guess in compressed form in SCR1
            SCALE = 1.0D0/SQRT(DFLOAT(NDEG))
            DO 250 II = 1,NDEG
*. Anti symmetric combination
              IF(INIDEG.EQ.-1) THEN
                SCR1(II) = (-1.0D0)**II * SCALE
*. Symmetric combination
              ELSE IF (INIDEG.EQ.1.OR.INIDEG.EQ.0) THEN
                SCR1(II) =  SCALE
              END IF
  250       CONTINUE
            IF(IDIAG.EQ.2) THEN
              JPACK = 1
            ELSE
              JPACK = 0
            END IF
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           WRITE(6,*) ' Initial guess modified '
C           ONE = 1.0D0
C           KELMNT = 17
C           CALL WRSVCD(LU1,LBLK,VEC1,KELMNT,ONE,1,NDIM,              
C    &           LUDIA,JPACK)
*
            CALL WRSVCD(LU1,LBLK,VEC1,ISCR1(IBASE),SCR1,NDEG,NDIM,
     &           LUDIA,JPACK)
            IBASE = IBASE + NDEG
  100     CONTINUE
        ELSE
* =====================================
*. Initial approximations are in PEIGVC
* =====================================
          CALL REWINO(LU1)
          IF(IDIAG.EQ.2) THEN
            JPACK = 1
          ELSE
            JPACK = 0
          END IF
          DO 1984 IROOT = 1, NROOT
            CALL WRSVCD(LU1,LBLK,VEC1,IPNTR,
     &           PEIGVC((IROOT-1)*NPRDET+1),NPRDET,NDIM,LUDIA,JPACK)
 1984     CONTINUE
          CALL MEMCHK2('AFWRSV')
        END IF
      END IF
*
* ======================================================
* Initial CI vectors are already on file LU1, do nothing
* ======================================================
*
      IF( INICI .LT. 0 ) THEN
*. Vectors assumed already in LU1
        IF(IPRT.GT. 1 )
     &  WRITE(6,*)' Initial CI vector assumed in place '
      END IF
*
*                 ========================
* 2:                  Diagonalization
*                 ========================
*
      CALL QENTER('CIEIG')
* Inverse iteration modified Davidsom with 2 vectorsin core
      IF(IPRT .GE. 5 ) THEN
         WRITE(6,*)
         WRITE(6,'(A,I3)')
     &   '  Number of roots to be converged..  ',NROOT
         WRITE(6,*)
C        WRITE(6,'(A,I3)')
C    &   '  Largest allowed number of vectors..',MAXVEC
         WRITE(6,*)
         WRITE(6,'(A,I3)')
     &   '  Allowed number of CI iterations  ..',MXCIIT
      END IF
*
      KRNRM = 1
      KEIG = KRNRM + MXCIIT*NROOT
      KFIN = KEIG  + MXCIIT*NROOT
      KAPROJ = KFIN + NROOT
      KAVEC = KAPROJ + MAXVEC*(MAXVEC+1)/2
      KWORK = KAVEC + MAXVEC ** 2
      KLFREE = KWORK + MAXVEC*(MAXVEC+1)
*
      KLRTCNV = KLFREE  
      KLFREE = KLRTCNV + NROOT
*
      IF( IPRT .GE. 100 ) THEN
        WRITE(6,*) ' KRNRM KEIG  KFIN  KAPROJ KAVEC KWORK KLFREE: '
        WRITE(6,'(7I5)')  KRNRM,KEIG,KFIN,KAPROJ,KAVEC,KWORK,KLFREE
      END IF
      IF( KLFREE-1 .GT. LLWRK) THEN
           WRITE(6,'(A,2I5)' )
     &     ' Not enough memory in CIEIG5: neeeded and available ',
     &     KLFREE-1, LLWRK
           WRITE(6,'(A,2I5)' )
     &     ' Increase parameter LLWRK in CIEIG5 to   ', KLFREE-1
           STOP ' insufficient memory in cieig5 '
       END IF
*
       IF(IUSEH0P.EQ.1) THEN
         WRITE(6,*) ' Special routine for H0 with projection '
         WRITE(6,*) ' Special routine for H0 with projection '
         WRITE(6,*) ' Special routine for H0 with projection '
         WRITE(6,*) ' Special routine for H0 with projection '
         WRITE(6,*) ' Special routine for H0 with projection '
         CALL MICDV4_H0LVP(VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT     ,MXCIIT,NDIM,LU3,LU4,LU5,LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &              THRES_E)
       ELSE IF (MPORENP_E.EQ.2) THEN
         WRITE(6,*) ' Special routine for EN-Lambda calc   '
         WRITE(6,*) ' Special routine for EN-Lambda calc   '
         WRITE(6,*) ' Special routine for EN-Lambda calc   '
         WRITE(6,*) ' Special routine for EN-Lambda calc   '
         WRITE(6,*) ' Special routine for EN-Lambda calc   '
         CALL MICDV4_ENLMD(MV7,VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT     ,MXCIIT,NDIM,LU3,LU4,LU5,LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &              THRES_E)
       ELSE
       IF(IDIAG.EQ.1.AND.ICISTR.EQ.1) THEN
*. Routine using two complete vectors in core
         IOLSEN = 1
         IPICO = 0
         IF(MXCIIT.NE.0) THEN
         CALL MINDV4(MV7,VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT     ,MXCIIT,NDIM,LU3,LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,EIGSHF,
     &              IOLSEN,IPICO,CONVER,RNRM_CNV,IROOT_SEL)
         ELSE 
*. No iterations, set energy to 0
           DO IROOT = 1, NROOT
             EROOT(IROOT) = 0.0D0
           END DO
         END IF
*. Routine with normal H0 operator
       ELSE IF(IDIAG.EQ.1.AND.ICISTR.GE.2) THEN
*
*. Routines using two blocks in core and numerous vectors
*
        IF(NROOT.EQ.1.AND.MAXVEC.EQ.2) THEN
*. Special routine for one root, 2 vectors
         CALL MICDV5(MV7,VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT     ,MXCIIT,NDIM,LU3,LU4,LU5,LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &              THRES_E,CONVER,RNRM_CNV)
        ELSE IF ( MAXVEC.LE.3*NROOT) THEN 
         CALL MICDV6(MV7,VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT,MXCIIT,NDIM,LU3,LU4,LU5,LU6,LU7,LU8,
     &              LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &              THRES_E,IROOT_SEL,NCNV_RT,SCR1(KLRTCNV),
     &              IPRECOND,CONVER,RNRM_CNV,ISBSPPR_ACT)
        ELSE IF(MAXVEC.GT.3*NROOT) THEN
*. General diag
C        WRITE(6,*) ' MICDV4 called '
         CALL MICDV4(MV7,VEC1,VEC2,LU1,LU2,SCR1(KRNRM),SCR1(KEIG),
     &              EROOT     ,MXCIIT,NDIM,LU3,LU4,LU5,LUDIA,NROOT,
     &              MAXVEC,NROOT,SCR1(KAPROJ),SCR1(KAVEC),
     &              SCR1(KWORK) ,IPRT,
     &              NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &              THRES_E,CONVER,RNRM_CNV,ISBSPPR_ACT)
         END IF
       ELSE IF(IDIAG.EQ.2)  THEN
*.Routine using two vector segments and three files
         NSUB = 0
         WRITE(6,*) ' ISKIPEI set to zero in CIEIG5'
         ISKIPEI = 0
         CALL PICO4(VEC1,VEC2,LU1,LU2,LU3,LU4,SCR1(KRNRM),
     &        SCR1(KEIG),
     &        EROOT,MXCIIT,NBATCH,
     &        ICBLBT,ICLEBT,ICLBLK,ICI1BT,ICBLOCK,
     &        ISBLBT,ISLEBT,ISLBLK,ISI1BT,ISBLOCK,
     &        IPRT,
     &        NPRDET,H0,IPNTR,NP1,NP2,NQ,H0SCR,LBLK,EIGSHF,
     &        THRES_E,E_THRE,C_THRE,E_CONV,C_CONV,ICLSSEL,
     &        IBLK_TO_CLS,NCLS,CLS_C,CLS_E,CLS_CT,CLS_ET,
     &        CLS_A,ICLS_L,RCLS_L,BLKS_A,CLS_DEL,CLS_DELT,
     &        CLS_GAMMA,CLS_GAMMAT,
     &        ISKIPEI,I2BLK,VEC3,
     &        ICLS_A2,MXLNG,IBASSPC,EBASC,CBASC,NSPC,
     &        MULSPC,IPAT,LPAT,ISPC,CONVER,RNRM_CNV)


       END IF
       END IF
*
       ENOT = SCR1(KFIN)
       CALL QEXIT('CIEIG')
*
      RETURN
      END
      SUBROUTINE COPMT2(AIN,AOUT,NINR,NINC,NOUTR,NOUTC,IZERO)
*
* Copy matrix AIN to AOUT . Dimensions can differ 
*
* DANGEROUS IF DIM of AOUT is smaller than of AIN !!!!
*
*
* If IZERO .ne. 0 , AOUT is zeroed  first
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION AIN(NINR,NINC)
      DIMENSION AOUT(NOUTR,NOUTC)
*
      IF(IZERO.NE.0) CALL SETVEC(AOUT,0.0D0,NOUTR*NOUTC)
      DO 100 J = 1, NINC
       CALL COPVEC(AIN(1,J),AOUT(1,J),NINR)
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE DEGVEC(VEC,NDIM,NDGVL,IDEG)
*
* A vector VEC is given with elements in ascending order
* group elements in degenerate pairs
*
*=======
* Input
*=======
* VEC: input vector
* NDIM: Number of elements in vec
*
*========
* Output
*========
* NDGVL: Number of degenerate values
* IDEG(I): Number of elements in VEC with degenerate value I
*
* Jeppe Olsen , April 1990
*
      IMPLICIT REAL*8           ( A-H,O-Z)
*.Input
      DIMENSION VEC(*)
*.Output
      DIMENSION IDEG(*)
*.Threshold for defining degenerency
      THRES = 1.0D-8
C?      write(6,*) ' Input vector to DEGVEC '
C?      call wrtmat(VEC,1,NDIM,1,NDIM)
      XDGVL = VEC(1)
      NDEG = 1
      NDGVL = 0
      DO 100 I = 2, NDIM
        IF(ABS(VEC(I)-XDGVL).LE.THRES) THEN
          NDEG = NDEG + 1
        ELSE
          NDGVL = NDGVL + 1
          IDEG(NDGVL) = NDEG
          XDGVL = VEC(I)
          NDEG = 1
        END IF
  100 CONTINUE
*. Last group
      NDGVL = NDGVL + 1
      IDEG(NDGVL) = NDEG
*
      NTEST = 0
      IF(NTEST .GT. 0 ) THEN
        WRITE(6,*) ' Output from DEGVEC '
        WRITE(6,*) ' ================== '
        WRITE(6,*)
        WRITE(6,*) ' Number of degenerate values ' ,NDGVL
        WRITE(6,*) ' Degenerencies of each value '
        CALL IWRTMA(IDEG,1,NDGVL,1,NDGVL)
      END IF
*
      RETURN
      END
      SUBROUTINE DETSTR(IDET,IASTR,IBSTR,NAEL,NBEL,
     &ISIGN,IWORK,IPRNT)
C
C A DETERMINANT,IDET,IS GIVEN AS A SET OF OCCUPIED SPIN ORBITALS,
C POSITIVE NUMBER INDICATES ALPHA ORBITAL AND NEGATIVE NUMBER
C INDICATES BETA ORBITAL .
C
C FIND CORRESPONDING ALPHA STRING AND BETA STRING ,
C AND DETERMINE SIGN NEEDED TO CHANGE DETERMINANT
C INTO PRODUCT OF ORDERED ALPHA STRING AND
C BETA STRING
C
C JEPPE OLSEN NOVEMBER 1988
*
* Two arguments (NEL, NOCOB) removed July 2011
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION IDET(*)
      DIMENSION IASTR(NAEL),IBSTR(NBEL)
      DIMENSION IWORK(*)
C required length of IWORK: NAEL + NBEL
C
      NTEST = 00
      NTEST = MAX(NTEST,IPRNT)
      NEL = NAEL + NBEL
C
C FIRST REORDER SPIN ORBITALS IN ASCENDING SEQUENCE
C THIS WILL AUTOMATICALLY SPLIT ALPHA AND BETASTRING
C
C     ORDSTR(IINST,IOUTST,NELMNT,ISIGN )
      CALL ORDSTR(IDET,IWORK,NEL,ISIGN,IPRNT)
C
C ALPHA STRING IS LAST NAEL ORBITALS
      CALL ICOPVE(IWORK(NBEL+1),IASTR,NAEL)
C
C BETA  STRING MUST BE COMPLETELY TURNED AROUND
      DO 10 IBEL = 1, NBEL
        IBSTR(IBEL) = -IWORK(NBEL+1-IBEL)
   10 CONTINUE
C SIGN CHANGE FOR SWITCH OF BETA ORBITALS
      NPERM = NBEL*(NBEL-1)/2 + NAEL*NBEL
      IF(MOD(NPERM,2).EQ.1) ISIGN = -ISIGN
COLD? ISIGN = ISIGN * (-1) ** (NBEL*(NBEL+1)/2)
C
      IF( NTEST .GE.10000) THEN
        WRITE(6,*) ' INPUT DETERMINANT '
        CALL IWRTMA(IDET,1,NEL,1,NEL)
        WRITE(6,*) ' CORRESPONDING ALPHA STRING '
        CALL IWRTMA(IASTR,1,NAEL,1,NAEL)
        WRITE(6,*) ' CORRESPONDING BETA STRING '
        CALL IWRTMA(IBSTR,1,NBEL,1,NBEL)
        WRITE(6,*) ' ISIGN FOR SWITCH ', ISIGN
      END IF
C
      RETURN
      END
      SUBROUTINE DGMM2 (AOUT,AIN,DIAG,IWAY,NRDIM,NCDIM)
C
C PRODUCT OF DIAGONAL MATRIX AND MATRIX:
C
C     IWAY = 1: AOUT(I,J) = DIAG(I)*AIN(I,J)
C     IWAY = 2: AOUT(I,J) = DIAG(J)*AIN(I,J)
C
      IMPLICIT REAL*8          (A-H,O-Z)
      DIMENSION AIN(NRDIM,NCDIM),DIAG(*)
      DIMENSION AOUT(NRDIM,NCDIM)
C
      IF ( IWAY .EQ. 1 ) THEN
         DO 100 J = 1, NCDIM
           CALL VVTOV(AIN(1,J),DIAG(1),AOUT(1,J),NRDIM)
  100    CONTINUE
      END IF
C
      IF( IWAY .EQ. 2 ) THEN
        DO 200 J = 1, NCDIM
          FACTOR = DIAG(J)
          CALL VECSUM(AOUT(1,J),AOUT(1,J),AIN(1,J),0.0D0,
     &                FACTOR,NRDIM)
  200   CONTINUE
      END IF
C
      NTEST = 00
      IF( NTEST .NE. 0 ) THEN
        WRITE(6,*) ' AIN DIAG AOUT  FROM DGMTMT '
        CALL WRTMAT(AIN ,NRDIM,NCDIM,NRDIM,NCDIM)
        IF(IWAY.EQ.1) THEN
        CALL WRTMAT(DIAG,1   ,NRDIM,1,NRDIM)
        ELSE
        CALL WRTMAT(DIAG,1   ,NCDIM,1,NCDIM)
        END IF
        CALL WRTMAT(AOUT,NRDIM,NCDIM,NRDIM,NCDIM)
      END IF
C
      RETURN
      END
      SUBROUTINE DISKUN
*
* Assign logical unit numbers for LUCIA:
*
* All file with some kind of input information  :  10 - 19
* All files containing final results            :  90 - 99
* Scratch files                                 :  30 - 50
* Internal files (retained through routines)    :  20 - 29
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'clunit.inc'
* =========================
* Standard input and output
* =========================
*. Input file
      LUIN = 5
*. Output file
      LUOUT = 6
* NEW * NEW * NEW *
* let a file-handler take care of unit-numbers
* the aim is to get rid of all fixed assignments below
      NTEST = 00
      CALL FH_INIT(IDUM,0,NTEST)

* =================
* Input information
* =================
* Input file containing MO-AO transformation matrix
      LUMOIN = IGETUNIT(12)
*. Input file for CI-vectors
*. restart from file 21 is assumed
*. Input , two electron integrals - MOLCAS
      LU2INT = IGETUNIT(13)
*. Input , one electron integrals - MOLCAS
      LU1INT = IGETUNIT(14)
*. Input , property one-electron integral files
      LUPRP  = IGETUNIT(15)
*. Sirius interface file
      LUSIR1 = IGETUNIT(16)
*. File containing additional states for transition densities - or for 
*. restart of CI coefficients in IC calculations 
      LUEXC = IGETUNIT(17)
* =================
* Internal files
* =================
*. CI diagonal
      LUDIA = IGETUNIT(20)
      CALL DANAME(LUDIA,'CIDIA')
*. CI vector
      LUC = IGETUNIT(21)
      CALL DANAME(LUC,'LUCVECT')
*. Sigma vector file
      LUHC = IGETUNIT(22)
      CALL DANAME(LUHC,'HCFILE')
*** the following unit are deactivated
*. File collecting CC correction vectors, used for DIIS etc
      LU_CCVEC = -1000
*. File containing approximations to the CC solutions
      LU_CCVECT = IGETUNIT(23)
*. File containing CC vector functions for the CC vectors on LU_CCVECT
      LU_CCVECF = IGETUNIT(24)
*. File containing last CC coefficients
      LU_CCVECL = IGETUNIT(25)
*. File containing Last CC vector function
      LU_CCVECFL = IGETUNIT(26)
* =================
* Scratch files
* =================
      LUSC1 = IGETUNIT(31)
      LUSC2 = IGETUNIT(32)
      LUSC3 = IGETUNIT(33)
      CALL DANAME(LUSC1,'LUSC1')
      CALL DANAME(LUSC2,'LUSC2')
      CALL DANAME(LUSC3,'LUSC3')
*. Scratch space for subspace handling 
      LUSC34 = IGETUNIT(34)
      LUSC35 = IGETUNIT(35)
      LUSC36 = IGETUNIT(36)
      LUSC37 = IGETUNIT(37)
      LUSC38 = IGETUNIT(38)
      LUSC39 = IGETUNIT(39)
      LUSC40 = IGETUNIT(40)
      LUSC41 = IGETUNIT(41)
      LUSC43 = IGETUNIT(43)
      LUSC44 = IGETUNIT(44)
      LUSC45 = IGETUNIT(45)
      CALL DANAME(LUSC34,'LUSC34')
      CALL DANAME(LUSC35,'LUSC35')
      CALL DANAME(LUSC36,'LUSC36')
      CALL DANAME(LUSC37,'LUSC37')
      CALL DANAME(LUSC38,'LUSC38')
      CALL DANAME(LUSC39,'LUSC39')
      CALL DANAME(LUSC40,'LUSC40')
      CALL DANAME(LUSC41,'LUSC41')
      CALL DANAME(LUSC43,'LUSC43')
      CALL DANAME(LUSC45,'LUSC45')
*.
      LUSC51 = IGETUNIT(51)
      LUSC52 = IGETUNIT(52)
      LUSC53 = IGETUNIT(53)
      LUSC54 = IGETUNIT(54)
*. Use the LUSC6* as scratch files under command of mini-file manager
      LUSC60 = IGETUNIT(60)
      LUSC61 = IGETUNIT(61)
      LUSC62 = IGETUNIT(62)
      LUSC63 = IGETUNIT(63)
      LUSC64 = IGETUNIT(64)
      LUSC65 = IGETUNIT(65)
      LUSC66 = IGETUNIT(66)
      LUSC67 = IGETUNIT(67)
      LUSC68 = IGETUNIT(68)
      LUSC69 = IGETUNIT(69)
*. Tell the mini-manager, about the above files, and that they
*  are available
      LUSUPSCR_IB = 60
      DO IFIL = 1, MXPNSCRFIL
        LU_SUPSCR(IFIL) = LUSUPSCR_IB - 1 + IFIL
        ISTAT_SUPSCR(IFIL) = 0
      END DO

* =================
* Output files
* =================
*. output file for CI-vectors
*. Not in use
      LUCIVO = IGETUNIT(98)
*. Natural orbitals in terms of input orbitals
*.
      LUMOUT = IGETUNIT(19)
      CALL DANAME(LUMOUT,'LUMOUT')
*. Dumping 1- and 2- electron integrals in formatted form
C     LU90  = IGETUNIT(90)
      LU90 = 90
*. Dumping symmmetry info, MO-AO expansion matrix and property integrals
C     LU91 = IGETUNIT(91)
      LU91 = 91
*. CC amplitudes in formatted form
      LU_CCAMP = IGETUNIT(92)
*. Result of CI=> CC conversion
      LU_CC_FROM_CI = IGETUNIT(93)
*. Excitation operators, all symmetries 
      LU_CCEXC_OP = IGETUNIT(94)
*. File for saving Metric and approx Jacobian in MRCC calculations
      LU_SJ = IGETUNIT(95)
      RETURN
      END
      SUBROUTINE DXTYP(NDXTP,ITYP,JTYP,KTYP,LTYP,LEL1,LEL3,REL1,REL3)
*
* Obtain types of I,J,K,l so
* <L!a+I a+K a L a J!R> is nonvanishing
* only combinations with type(I) .ge. type(K) and type(L).ge.type(J)
* are included
*
      INTEGER REL1,REL3
      INTEGER ITYP(6),JTYP(6),KTYP(6),LTYP(6)
*
*. To get rid of annoying and incorrect compiler warnings
      I1 = 0
      I3 = 0
      IK1 = 0
      IK3 = 0
      IKL1 = 0
      IKL3 = 0
      IKLJ1 = 0
      IKLJ3 = 0
*
      NDXTP = 0
      DO 400 ITP = 1, 3
        IF(ITP.EQ.1) THEN
          I1 = 1
          I3 = 0
        ELSE IF(ITP.EQ.2) THEN
          I1 = 0
          I3 = 0
        ELSE IF(ITP.EQ.3) THEN
          I1 = 0
          I3 = 1
        END IF
        DO 300 KTP = 1, ITP
          IF(KTP.EQ.1) THEN
            IK1 = I1+1
            IK3 = I3
          ELSE IF(KTP.EQ.2) THEN
            IK1 = I1
            IK3 = I3
          ELSE IF(KTP.EQ.3) THEN
            IK1 = I1
            IK3 = I3+1
          END IF
          IF(LEL1-IK1.LT.0) GOTO 300
          IF(LEL3-IK3.LT.0) GOTO 300
          DO 200 LTP = 1,3
            IF(LTP.EQ.1) THEN
              IKL1 = IK1-1
              IKL3 = IK3
            ELSE IF(LTP.EQ.2) THEN
              IKL1 = IK1
              IKL3 = IK3
            ELSE IF(LTP.EQ.3) THEN
              IKL1 = IK1
              IKL3 = IK3-1
            END IF
            DO 100 JTP = 1, 3
              IF(JTP.EQ.1) THEN
                IKLJ1 = IKL1-1
                IKLJ3 = IKL3
              ELSE IF(JTP.EQ.2) THEN
                IKLJ1 = IKL1
                IKLJ3 = IKL3
              ELSE IF(JTP.EQ.3) THEN
                IKLJ1 = IKL1
                IKLJ3 = IKL3-1
              END IF
              IF(IKLJ1+REL1.EQ.LEL1.AND.IKLJ3+REL3.EQ.LEL3) THEN
                NDXTP = NDXTP + 1
                ITYP(NDXTP) = ITP
                KTYP(NDXTP) = KTP
                LTYP(NDXTP) = LTP
                JTYP(NDXTP) = JTP
              END IF
  100       CONTINUE
  200     CONTINUE
  300   CONTINUE
  400 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,'(A,4I4)')
     &  ' Double excitations connecting LEL1,LEL3,LEL1,LEL3 ',
     &    LEL1,LEL3,REL1,REL3
        WRITE(6,*) '  ITYP KTYP LTYP JTYP '
        WRITE(6,*) '  ===================='
        DO 10 IDX = 1,NDXTP
          WRITE(6,'(1H ,5I5)')ITYP(IDX),KTYP(IDX),LTYP(IDX),JTYP(IDX)
   10   CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE ERES(LUC,LUHC,VEC1,VEC2,LBLK,CC,CHC,CHHC,
     &                NSUB,ISUB,CSUB,HCSUB,IPRT)
*
* Calculate terms needed for evaluating the energy  and
* the residual norm through a single loop through LU, LUHC
*
*. Output is 
* ==========
* CC = <C!C>
* CHC = <C!H!C>
* CHHC = <C!HH!C>
*
* If NSUB .NE. 0 the elements of C and HC belonging to the
* explicit subspace are obtained and stored in CSUB,HCSUB
*
      IMPLICIT REAL*8           (A-H,O-Z)
      REAL*8 INPROD
*
      DIMENSION VEC1(*),VEC2(*),ISUB(*),CSUB(*),HCSUB(*)
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRT)
*
      IF(NTEST.GE.5)
     ?WRITE(6,*) ' LUC LUHC LBLK in ERES ',LUC,LUHC,LBLK
      CALL REWINE(LUC,LBLK)
      CALL REWINE(LUHC,LBLK)
*
      IBASE = 1
      IEFF = 1
      CHC = 0.0D0
      CHHC = 0.0D0
      CC = 0.0D0
*. LOOP OVER BLOCKS
 1000 CONTINUE
        CALL NEXREC(LUC,LBLK,VEC1,IEND,LENGTH)
        CALL NEXREC(LUHC,LBLK,VEC2,IEND,LENGTH)
        IF(NTEST.GE.5)
     &  WRITE(6,*) ' Number of elements in block ',LENGTH
        IF(IEND.EQ.0) THEN
          CC = CC + INPROD(VEC1,VEC1,LENGTH)
          CHC = CHC + INPROD(VEC1,VEC2,LENGTH)
          CHHC = CHHC + INPROD(VEC2,VEC2,LENGTH)
*
          IF(NSUB.NE.0) THEN
            IFIRST = IBASE
            ILAST = IFIRST + LENGTH - 1
            DO 100 JSUB = 1, NSUB
             IF(ISUB(JSUB).GE.IFIRST.AND.ISUB(JSUB).LE.ILAST)THEN
               CSUB(JSUB) = VEC1(ISUB(JSUB)-IBASE+1)
               HCSUB(JSUB) = VEC2(ISUB(JSUB)-IBASE+1)
             END IF
  100       CONTINUE
          END IF
*
         IBASE = IBASE + LENGTH
      GOTO 1000
        END IF
*
      IF(NTEST.GE.5) THEN
       WRITE(6,*) ' CC CHC CHHC FROM ERES ',CC,CHC,CHHC
      END IF
*
      RETURN
      END
      SUBROUTINE FNDMN3(VEC,NDIM,MXELMN,IPLACE,NELMN,
     &                  IPRT,THRES )
C
C FIND MXELMN/NELMN LOWEST ELEMENTS IN VEC .
C NELMN IS THE LARGEST NUMBER LOWER THAN MXELMN THAT DOES NOT
C SPLIT DEGENERATE PAIRS
C ORIGINAL PLACES OF THE LOWEST ELEMENTS ARE STORED IN IPLACE
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION VEC(NDIM),IPLACE(*)
C
C
C. FIRST OCCURANCE OF LOWEST ELEMNT AND LARGEST ELEMENT
      XMIN = VEC(1)
      XMAX = VEC(1)
      IMIN = 1
      IMAX = 1
      DO 100 I = 2,NDIM
        IF( VEC(I) .GT. XMAX) THEN
           XMAX = VEC(I)
           IMAX = I
        END IF
  100 CONTINUE
      DO 101 I = 2,NDIM
        IF( VEC(I) .LT. XMIN) THEN
           XMIN = VEC(I)
           IMIN = I
        END IF
  101 CONTINUE
C
      IF(IPRT .GE.2 ) WRITE(6,*) ' LOWEST VALUE AND PLACE ',XMIN,IMIN
      IF(IPRT .GE.2 ) WRITE(6,*) ' HIGHST VALUE AND PLACE ',XMAX,IMAX
      IPLACE(1) = IMIN
C
      NDEG = 1
      ITOP = MIN(NDIM,MXELMN+1)
      DO 200 IELMNT = 2,ITOP
        XMINPR = XMIN
        IMINPR = IMIN
        XMIN = XMAX
        IMIN = IMAX
        DO 150 I = 1,NDIM
          IF(VEC(I).LT.XMIN.AND.VEC(I).GT.XMINPR .OR.
     &       VEC(I).EQ.XMINPR .AND. I .GT. IMINPR ) THEN
            IMIN = I
            XMIN = VEC(I)
            IF(XMIN .EQ. XMINPR ) GOTO 151
          END IF
  150   CONTINUE
  151   CONTINUE
        IF(XMIN-XMINPR .LT. THRES )THEN
         NDEG = NDEG + 1
        ELSE
         NDEG = 1
        END IF
C
        IF( IELMNT .LE. MXELMN ) THEN
          IPLACE(IELMNT) = IMIN
          IF(IPRT .GE.3 ) WRITE(6,*) 'IELMNT XMIN IMIN NDEG ',
     &    IELMNT,XMIN,IMIN,NDEG
        END IF
  200   CONTINUE
C
C CHECK DEGENERENCY ON LAST VALUE
            IF(MXELMN .LT. NDIM ) THEN
              NELMN = MXELMN+1-NDEG
             ELSE
              NELMN = NDIM
             END IF
C?    WRITE(6,*) ' NUMBER OF ELEMENTS OBTAINED IN FNDMN3 ',NELMN
C
C
      IF( IPRT  .GE. 3 ) THEN
        WRITE(6,*) ' FROM FNDMN3: '
        WRITE(6,*) '   PLACES OF LOWEST ELEMENTS '
        CALL IWRTMA(IPLACE,1,NELMN ,1,NELMN )
      END IF
C
      IF( IPRT .GT. 0 ) THEN
       WRITE(6,*)
     & ' MIN AND MAX IN SELECTED SUPSPACE ',
     &   VEC(IPLACE(1)),VEC(IPLACE(NELMN))
      END IF
C
      RETURN
      END
      SUBROUTINE FNDMND(LU,LBLK,SEGMNT,NSUBMX,NSUB,ISCR,
     &SCR,ISCAT,SUBVAL,NTESTG)
*
* FIND NSUB LOWEST ELEMENTS OF VECTOR RESIDING ON FILE
* LU. ENSURE THAT NO DEGENERENCIES ARE SPLIT
*
*
* INPUT
*======
* LU: FILE WHERE VECTOR OF INTEREST IS RESIDING, REWINDED
* LBLK: DEFINES FILE STRUCTURE ON FILE LU
*        IF(LBLK.LT.0) THEN vector is supposed to be single record of length LBLK
* NSUBMX: LARGEST ALLOWED NUMBER OF SORTED ELEMENTS
*
*OUTPUT
*======
* NSUB: ACTUAL NUMBER OF ELEMENTS OBTAINED. CAN BE SMALLER
*       THAN NSUBMX IF THE LATST ELEMENT BELONGS TO A DEGENERATED
*       SET
*ISCAT: SCATTERING ARRAY , ISCAT(I) GIVED FULL ADRESS OF SORTED
*       ELEMENT I
*       SUBVAL: VALUE OF SORTED ELEMENTS
 
      IMPLICIT REAL*8           ( A-H,O-Z)
      DIMENSION SEGMNT(*), ISCAT(*),SUBVAL(*),SCR(*),ISCR(*)
*
      CALL REWINE(LU,-1)
*
*.LOOP OVER BLOCKS
*
C     write(6,*) ' FNDMND NSUBMX = ', NSUBMX
      NTESTL = 0
      NTEST = MAX(NTESTG,NTESTL)
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Output from FNDMND'
        WRITE(6,*) ' ==================='
        WRITE(6,*)
        WRITE(6,*) ' LBLK = ', LBLK
      END IF
*
      IBLK = 0
      IBASE = 1
      LSUB = 0
 1000 CONTINUE
        IF ( LBLK .GT. 0 ) THEN
          LBL = LBLK
        ELSE
          CALL IFRMDS(LBL,1,-1,LU)
        END IF
        IBLK = IBLK + 1
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' Info about block ',IBLK
          WRITE(6,*) ' Number of elements ', LBL
        END IF
        IF(LBL .GE. 0 ) THEN
          IF(LBLK .GE.0 ) THEN
            KBLK = LBL
          ELSE
            KBLK = -1
          END IF
           CALL FRMDSC(SEGMNT,LBL ,-1,LU,IMZERO,IAMPACK)
           IF(NTEST.GE.10000) THEN
             WRITE(6,*) ' Elements read in '
             CALL WRTMAT(SEGMNT,1,LBL,1,LBL)
           END IF
           IF(LBL .GE. 0 ) THEN
*. LOWEST ELEMENTS IN SEGMNT  ( ADD TO PREVIOUS LIST )
             MSUBMX = MIN(NSUBMX,LBL)
C     SUBROUTINE SORLOW(WRK,STVAL,ISTART,KZVAR,KEXSTV,JEXSTV,IPRT)
             IF(LBL.GE.1) THEN
               CALL SORLOW(SEGMNT,SCR(1+LSUB),ISCR(1+LSUB),LBL,
     &                     MSUBMX,MSUB,NTEST)
C              WRITE(6,*)
C    &       ' After SORLOW 1 , Scatter array for combined list '
C            call iwrtma(ISCR(1),1,LSUB+MSUB,1,LSUP+MSUP)
             ELSE
              MSUB = 0
             END IF
             DO 10 I = 1, MSUB
   10        ISCR(LSUB+I) = ISCR(LSUB+I) + IBASE - 1
C            Write(6,*) 
C    &       ' After 10 , Scatter array for combined list '
C            call iwrtma(ISCR,1,LSUB+MSUB,1,LSUP+MSUP)
* SORT COMBINED LIST
             MSUBMX = MIN(NSUBMX,LSUB+MSUB)
             IF(MSUBMX.GT.0) THEN
             CALL SORLOW(SCR,SUBVAL,ISCAT,LSUB+MSUB,MSUBMX,LLSUB,
     &         NTEST)
C              WRITE(6,*)
C    &       ' After SORLOW 2 , Scatter array for combined list '
C            call iwrtma(ISCR(1),1,LSUB+MSUB,1,LSUP+MSUP)
             ELSE
               LLSUB = 0
             END IF
             LSUB = LLSUB
             DO 20 I = 1, LSUB
               ISCR(I+2*NSUBMX) = ISCR(ISCAT(I))
   20        CONTINUE
*
             CALL ICOPVE(ISCR(1+2*NSUBMX),ISCR(1),LSUB)
             CALL COPVEC(SUBVAL,SCR,LSUB)
             IBASE = IBASE + LBL
             IF(NTEST .GE. 1000 ) THEN
               WRITE(6,*) ' Lowest elements and their original place '
               WRITE(6,*) ' Number of elements obtained ', LSUB
               CALL WRTMAT(SUBVAL,1,LSUB,1,LSUB)
               CALL IWRTMA(ISCR,1,LSUB,1,LSUB)
             END IF
          END IF
*
        END IF
*
      IF( LBL.GE. 0 .AND. LBLK .LE. 0) GOTO 1000
*
      NSUB = LSUB
      CALL ICOPVE(ISCR,ISCAT,NSUB)
      IF(NTEST .GE. 100) THEN
        WRITE(6,*) ' Lowest elements and their original place '
        WRITE(6,*) ' Number of elements obtained ', NSUB
       CALL WRTMAT(SUBVAL,1,NSUB,1,NSUB)
       CALL IWRTMA(ISCAT,1,NSUB,1,NSUB)
      END IF
*
      RETURN
      END
      SUBROUTINE GATVC2(IGAT,SGAT,VECIN,VECOUT,NDIM,FACTOR)
*
* VECOUT(I) = SGAT(I)*VECIN(IGAT(I))*FACTOR if IGAT(I) .NE.0
* VECOUT(I) = 0                             if IGAT(I) .EQ.0
*
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      DIMENSION IGAT(*),SGAT(*),VECIN(*)
*.Output
      DIMENSION VECOUT(*)
*
      DO 100 I = 1, NDIM
        IF(IGAT(I).EQ.0) THEN
          VECOUT(I) = 0.0D0
        ELSE
          VECOUT(I) = VECIN(IGAT(I))*SGAT(I)*FACTOR
        END IF
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE GDEPS(LUC,LUHC,LUDIA,ISUB,ESUB,VSUB,NSUB,SCR,
     &                  E,GAMMA,DELTA,EPSIL,VEC1,VEC2,VEC3,
     &                  HCSUB,CSUB,LBLK,IPRT)
*
* Construct DELTA = <C!(H0-1)**-1!C>
*           GAMMA = <C!(H0-E)**-1(H-E)!C>
*           EPSIL = NORM OF !(H0-E)!C>
*
*
* WITH C IN LU1, HC ON LUHC AND  DIAGONAL ON LUDIA
*
* MICRO VERSION , ONLY ASSUMED TO BE ABLE
* TO HOLD 3 SEGMENTS OF VECTORS
*
 
      IMPLICIT REAL*8           (A-H,O-Z)
      REAL*8 INPROD
      DIMENSION SCR(*),VEC1(*),VEC2(*),VEC3(*)
      DIMENSION ISUB(*),ESUB(*),VSUB(*)
      DIMENSION CSUB(*),HCSUB(*)
 
 
* ISUB IS ASSUMED ORDERED
*
      CALL REWINE(LUDIA,LBLK)
      CALL REWINE(LUC  ,LBLK)
      CALL REWINE(LUHC,LBLK)
      IBASE = 1
      IEFFC = 1
      IEFFHC = 1
*
      GAMMA = 0.0D0
      DELTA = 0.0D0
      EPSIL = 0.0D0
*===============================================
** CONTRIBUTIONS FROM SPACE WHERE H0 IS DIAGONAL
*===============================================
*
* LOOP OVER RECORDS ON LUC, LUHC, LUDIA
*
 1000 CONTINUE
C     NEXREC(LU,LBLK,REC,IEND,LENGTH)
*. NEXT RECORD OF C
        CALL NEXREC(LUC,LBLK,VEC1,IEND,LENGTH)
*. NEXT RECORD OF HNOT
        CALL NEXREC(LUDIA,LBLK,VEC2,IEND,LENGTH)
        IF(IEND.EQ.0) THEN
*
          DO 100 I = 1, LENGTH
            IF(IEFFC.LE.NSUB.AND.I .EQ. ISUB(IEFFC)-IBASE+1)THEN
* C  IN SUBSPACE
              CSUB(IEFFC)= VEC1(I)
              IEFFC = IEFFC + 1
            ELSE
              IF(ABS(E-VEC2(I)).GT.1.0D-10) THEN
                DIVIDE = VEC2(I) - E
              ELSE
                DIVIDE = 1.0D-10
              END IF
              DELTA = DELTA + VEC1(I) ** 2/DIVIDE
              EPSIL = EPSIL + (VEC1(I) * (VEC2(I) - E))**2
              VEC2(I) = VEC1(I)/DIVIDE
            END IF
  100     CONTINUE
*. NEXT RECORD OF HC
          CALL NEXREC(LUHC,LBLK,VEC1,IEND,LENGTH)
          DO 200 I = 1, LENGTH
            IF(IEFFHC.LE.NSUB.AND.I .EQ. ISUB(IEFFHC)-IBASE+1)THEN
*HC  IN SUBSPACE
              HCSUB(IEFFHC)= VEC1(I)
              IEFFHC = IEFFHC + 1
            ELSE
              GAMMA = GAMMA + VEC1(I)*VEC2(I)
            END IF
  200     CONTINUE
*
          IBASE = IBASE + LENGTH
      GOTO 1000
        END IF
*
** CONTRIBUTIONS FROM SUBSPACE
*
 
      KVEC1 = 1
      KVEC2 = KVEC1 + NSUB
*..(H0-E)**-1 C IN SUBSPACE
C     XDXTV(VECUT,VECIN,X,DIA,NDIM,SCR,SHIFT,IINV)
      CALL XDXTV(SCR(KVEC1),CSUB,VSUB,ESUB,NSUB,SCR(KVEC2),-E,1)
      GAMMA = GAMMA + INPROD(HCSUB,SCR(KVEC1),NSUB)
      DELTA = DELTA + INPROD(CSUB,SCR(KVEC1),NSUB)
*..(H0-E)*C IN SUBSPACE
      CALL XDXTV(SCR(KVEC1),CSUB,VSUB,ESUB,NSUB,SCR(KVEC2),-E,0)
      EPSIL = EPSIL + INPROD(SCR(KVEC1),SCR(KVEC1),NSUB)
*
      GAMMA = GAMMA - E * DELTA
      EPSIL = SQRT(EPSIL)
      NTEST = 1
      NTEST = MAX(NTEST,IPRT)
      IF( NTEST .GE.5 ) THEN
        WRITE(6,*) ' GAMMA DELTA EPSIL FROM DELGAM',GAMMA,DELTA,EPSIL
      END IF
*
      RETURN
      END
      SUBROUTINE GENSTR(NEL,NELMN1,NELMX1,NELMN3,NELMX3,
     &                  ISTASO,NOCTYP,NSMST,Z,LSTASO,
     &                  IREORD,STRING,IOC,IOTYP,IPRNT)
*
* Generate strings consisting of  NEL electrons fullfilling
*   1: Between NELMN1 AND NELMX1 electrons in the first NORB1 orbitals
*   2: Between NELMN3 AND NELMX3 electrons in the last  NORB3 orbitals
*
* In the present version the strings are directly ordered into
* symmetry and occupation type .
*
* Jeppe Olsen Winter of 1990
* ========
* Output:
* ========
* STRING(IEL,ISTRIN): Occupation of strings.
* IREORD            : Reordering array going from lexical
*                      order to symmetry and occupation type order.
*
      IMPLICIT REAL*8           ( A-H,O-Z)
*. Input
      DIMENSION ISTASO(NOCTYP,NSMST)
      INTEGER Z(NACOB,NEL)
*.Orbinp
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*
*.Output
      INTEGER STRING(NEL,*),IREORD(*)
*.Scratch arrays
      DIMENSION IOC(*),LSTASO(NOCTYP,NSMST)
*
      CALL ISETVC(LSTASO,0,NOCTYP*NSMST)
      NTEST0 = 0
      NTEST = MAX(NTEST0,IPRNT)
      IF( NTEST .GE. 10) THEN
        WRITE(6,*)  ' =============== '
        WRITE(6,*)  ' GENSTR speaking '
        WRITE(6,*)  ' =============== '
      END IF
*
      NSTRIN = 0
      IORB1F = 1
      IORB1L = IORB1F+NORB1-1
      IORB2F = IORB1L + 1
      IORB2L = IORB2F+NORB2-1
      IORB3F = IORB2L + 1
      IORB3L = IORB3F+NORB3-1
* Loop over possible partitionings between RAS1,RAS2,RAS3
      DO 1001 IEL1 = NELMX1,NELMN1,-1
      DO 1003 IEL3 = NELMN3,NELMX3, 1
       IF(IEL1.GT. NORB1 ) GOTO 1001
       IF(IEL3.GT. NORB3 ) GOTO 1003
       IEL2 = NEL - IEL1-IEL3
       IF(IEL2 .LT. 0 .OR. IEL2 .GT. NORB2 ) GOTO 1003
       IFRST1 = 1
* Loop over RAS 1 occupancies
  901  CONTINUE
         IF( IEL1 .NE. 0 ) THEN
           IF(IFRST1.EQ.1) THEN
            CALL ISTVC2(IOC(1),0,1,IEL1)
            IFRST1 = 0
           ELSE
             CALL NXTORD(IOC,IEL1,IORB1F,IORB1L,NONEW1)
             IF(NONEW1 .EQ. 1 ) GOTO 1003
           END IF
         END IF
         IF( NTEST .GE. 500) THEN
           WRITE(6,*) ' RAS 1 string '
           CALL IWRTMA(IOC,1,IEL1,1,IEL1)
         END IF
         IFRST2 = 1
         IFRST3 = 1
* Loop over RAS 2 occupancies
  902    CONTINUE
           IF( IEL2 .NE. 0 ) THEN
             IF(IFRST2.EQ.1) THEN
              CALL ISTVC2(IOC(IEL1+1),IORB2F-1,1,IEL2)
              IFRST2 = 0
             ELSE
               CALL NXTORD(IOC(IEL1+1),IEL2,IORB2F,IORB2L,NONEW2)
               IF(NONEW2 .EQ. 1 ) THEN
                 IF(IEL1 .NE. 0 ) GOTO 901
                 IF(IEL1 .EQ. 0 ) GOTO 1003
               END IF
             END IF
           END IF
           IF( NTEST .GE. 500) THEN
             WRITE(6,*) ' RAS 1 2 string '
             CALL IWRTMA(IOC,1,IEL1+IEL2,1,IEL1+IEL2)
           END IF
           IFRST3 = 1
* Loop over RAS 3 occupancies
  903      CONTINUE
             IF( IEL3 .NE. 0 ) THEN
               IF(IFRST3.EQ.1) THEN
                CALL ISTVC2(IOC(IEL1+IEL2+1),IORB3F-1,1,IEL3)
                IFRST3 = 0
               ELSE
                 CALL NXTORD(IOC(IEL1+IEL2+1),
     &           IEL3,IORB3F,IORB3L,NONEW3)
                 IF(NONEW3 .EQ. 1 ) THEN
                   IF(IEL2 .NE. 0 ) GOTO 902
                   IF(IEL1 .NE. 0 ) GOTO 901
                   GOTO 1003
                 END IF
               END IF
             END IF
             IF( NTEST .GE. 500 ) THEN
               WRITE(6,*) ' RAS 1 2 3 string '
               CALL IWRTMA(IOC,1,NEL,1,NEL)
             END IF
* Next string has been constructed , Enlist it !.
             NSTRIN = NSTRIN + 1
*. Symmetry
*                   ISYMST(STRING,NEL)
             ISYM = ISYMST(IOC,NEL)
*. Occupation type
             ITYP = IOCTP2(IOC,NEL,IOTYP)
*
             IF(ITYP.NE.0) THEN
               LSTASO(ITYP,ISYM) = LSTASO(ITYP,ISYM)+ 1
C                      ISTRNM(IOCC,NACTOB,NEL,Z,NEWORD,IREORD)
               LEXCI = ISTRNM(IOC,NACOB,NEL,Z,IREORD,0)
               LACTU = ISTASO(ITYP,ISYM)-1+LSTASO(ITYP,ISYM)
               IREORD(LEXCI) = LACTU
               IF(NTEST.GT.10) WRITE(6,*) ' LEXCI,LACTU',
     &         LEXCI,LACTU
               CALL ICOPVE(IOC,STRING(1,LACTU),NEL)
             END IF
*
           IF( IEL3 .NE. 0 ) GOTO 903
           IF( IEL3 .EQ. 0 .AND. IEL2 .NE. 0 ) GOTO 902
           IF( IEL3 .EQ. 0 .AND. IEL2 .EQ. 0 .AND. IEL1 .NE. 0)
     &     GOTO 901
 1003 CONTINUE
 1001 CONTINUE
*
      IF(NTEST.GE.1 ) THEN
        WRITE(6,*) ' Number of strings generated   ', NSTRIN
      END IF
      IF(NTEST.GE.10)  THEN
        IF(NTEST.GE.100) THEN
          NPR = NSTRIN
        ELSE
          NPR = MIN(NSTRIN,50)
        END IF
        WRITE(6,*) ' Strings generated '
        WRITE(6,*) ' =================='
        ISTRIN = 0
        DO 100 ISYM = 1, NSMST
        DO 100 ITYP = 1,NOCTYP
          LSTRIN = MIN(LSTASO(ITYP,ISYM),NPR-ISTRIN)
          IF(LSTRIN.GT.0) THEN
            WRITE(6,*) ' Strings of type and symmetry ',ITYP,ISYM
            DO 90 KSTRIN = 1,LSTRIN
              ISTRIN = ISTRIN + 1
              WRITE(6,'(2X,I4,8X,(10I5))')
     &        ISTRIN,(STRING(IEL,ISTRIN),IEL = 1,NEL)
   90       CONTINUE
          END IF
  100   CONTINUE
*
        WRITE(6,*) ' Array giving actual place from lexical place'
        WRITE(6,*) ' ============================================'
        CALL IWRTMA(IREORD,1,NPR,1,NPR)
      END IF
 
      RETURN
      END
      Subroutine GetH0(H)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Load one electron integrals                                      *
*     File assumed opened by GETOBS                                    *
*                                                                      *
*     Calling parameters:                                              *
*     H   : core Hamiltonian matrix                                    *
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Implicit Real*8 (A-H,O-Z)
*
      Parameter( LuOne = 14)
*
      Dimension H(*)
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
      INTEGER*8 iToc,nBas,nOrb,nFro,nDel,Nsym
COLD  Character*4 Name(2,400)
*----------------------------------------------------------------------*
*----------------------------------------------------------------------*
*     Determine the number of integrals (symmetry blocked)             *
*----------------------------------------------------------------------*
      NorbTT=0
      Do iSym=1,nSym
        NorbTT=NorbTT+Norb(iSym)*(Norb(iSym)+1)/2
      End Do
*----------------------------------------------------------------------*
*     Load the core Hamiltonian matrix                                 *
*----------------------------------------------------------------------*
      iDisk=iToc(2)
      write(6,*) '********************************'
      write(6,*) '>>>>> In GetH0 <<<<<'
      write(6,*) 'nSym', nSym
      write(6,*) 'Norb(iSym)', (Norb(i),i=1,nSym)
      write(6,*) 'iDisk',iDisk
      write(6,*) 'NorbTT',NorbTT
      write(6,*) '********************************'
      Call dDaFile(LuOne,2,H,NorbTT,iDisk)
      write(6,*) 'Core Hamiltonian matrix (symmetry blocked)'
      iPoint=0
      DO iSym=1,nSym
        IF ( nOrb(iSym).GT.0 ) THEN
          WRITE(6,'(6X,A,I2)')' symmetry species:',iSym
          CALL TRIPRT(' ',' ',H(1+iPoint),nOrb(iSym))
          iPoint=iPoint+nOrb(iSym)*(nOrb(iSym)+1)/2
        END IF
      END DO

*GLMJ      Call DaFile(LuOne,2,H,2*NorbTT,iDisk)
*----------------------------------------------------------------------*
*     Terminate procedure                                              *
*----------------------------------------------------------------------*
      Return
      End
      Subroutine GetH0old(H,Ecore)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Load one electron integrals                                      *
*                                                                      *
*     Calling parameters:                                              *
*     H   : core Hamiltonian matrix                                    *
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Implicit Real*8 (A-H,O-Z)
*
      Parameter( LuOne = 14)
*
      Dimension H(*)
      Dimension IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8)
      Character*4 Name(2,400)
*----------------------------------------------------------------------*
*     Start procedure:                                                 *
*     open the transformed one-electron integral file                  *
*----------------------------------------------------------------------*
      Call DaName(LuOne,'TRAONE')
*----------------------------------------------------------------------*
*     Set up the scatter/gather list and read table of contents        *
*----------------------------------------------------------------------*
      iDisk=0
      Call GSList(IOList,8,iToc,64,Ecore,2,nSym,1,
     &            nBas,8,nOrb,8,nFro,8,nDel,8,Name,800)
      Call DaFile(LuOne,4,IOList,iDum,iDisk)
*----------------------------------------------------------------------*
*     Determine the number of integrals (symmetry blocked)             *
*----------------------------------------------------------------------*
      NorbTT=0
      Do iSym=1,nSym
        NorbTT=NorbTT+Norb(iSym)*(Norb(iSym)+1)/2
      End Do
*----------------------------------------------------------------------*
*     Load the core Hamiltonian matrix                                 *
*----------------------------------------------------------------------*
      iDisk=iToc(2)
      Call DaFile(LuOne,2,H,2*NorbTT,iDisk)
*----------------------------------------------------------------------*
*     Terminate procedure                                              *
*----------------------------------------------------------------------*
      Return
      End
*
      Subroutine GetH0S(H,NTORB)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Obtain one electron integrals                                    *
*     SIRIUS interface                                                 *
*                                                                      *
*     Calling parameters:                                              *
*     H   : core Hamiltonian matrix                                    *
*                                                                      *
*****  Author: Unknown                                *****************
*
c      Implicit Real*8 (A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
*
      Parameter (LUONE = 19)
*
      Dimension H(*)
*
*
      IDUM = 1
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'GETH0S')
*. Square form of H0
COLD  CALL MEMMAN(KLH0,NTORB**2,'ADDL  ',2,'KLH0  ')
*. Allocate scratch memory
      LSCR = NTORB + 7*NTORB**2
      CALL MEMMAN(KLSCR,LSCR,'ADDL  ',2,'H0SCR   ')
      CALL MEMMAN(KLISCR,NTORB,'ADDL  ',1,'H0ISCR')
*. Get one body matrix in MO basis
      CALL INFSIR(int_mb(KLISCR),dbl_mb(KLSCR),LSCR)
C                                                            
C----------------------------------------                    
C     Read in the one electron integrals.                    
C----------------------------------------                  
C                                    
      OPEN (LUONE,STATUS='UNKNOWN',FORM='UNFORMATTED',   
     *      FILE='MOONEINT')         
      READ(LUONE) NCMOT,(H(I),I=1,NCMOT)                    
      CLOSE (LUONE,STATUS='DELETE')                             
*
      NTEST = 0
      IF( NTEST .GE. 10 ) THEN
        WRITE(6,*) ' ====================================='
        WRITE(6,*) ' One electron MO-integrals from GETH0S'
        WRITE(6,*) ' ====================================='
        WRITE(6,*)
        CALL WRTMAT(H,1,NCMOT,1,NCMOT)        
      END IF
C
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'GETH0S')
      Return
      End
      SUBROUTINE GETH1(H,ISM,ITP,JSM,JTP)
*
* One-electron integrals over orbitals belonging to
* given OS class
*
*
* The orbital symmetries  are used to obtain the total
* symmetry of the one-electron integrals.
* It is therefore assumed that ISM, JSM represents a correct symmetry block
* of the integrals
*
* Jeppe Olsen, Version of fall 97
*              Summer of 98: CC options added
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
*.Global pointers
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'intform.inc'
*.Output
      DIMENSION H(*)
*
      NI = NOBPTS_GN(ITP,ISM)
      NJ = NOBPTS_GN(JTP,JSM)
*
      IJ = 0
      DO J = 1, NJ
        DO I = 1, NI
          IJ = IJ+1
          H(IJ) = GETH1E(I,ITP,ISM,J,JTP,JSM)
        END DO
      END DO
*
      NTEST = 00
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' One-electron integral block from GETH1'
        WRITE(6,*) ' ======================================'
        WRITE(6,*) ' IH1FORM = ', IH1FORM
        WRITE(6,*) ' H1 for itp ism jtp jsm ',ITP,ISM,JTP,JSM
        CALL WRTMAT(H,NI,NJ,NI,NJ)
      END IF
*
      RETURN
      END
      FUNCTION GETH1EX(IORB,ITP,ISM,JORB,JTP,JSM,H)
*
* One-electron integral for active
* orbitals (IORB,ITP,ISM),(JORB,JTP,JSM)
*
* The orbital symmetries are used to obtain the
* total symmetry of the operator
*
* Version where one-electron integrals H come through argument list !
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'multd2h.inc'
      INCLUDE 'intform.inc'
      INCLUDE 'oper.inc'
      DIMENSION H(*)
*
C?    WRITE(6,*) ' I_USE_SIMTRH, IH1FORM ', I_USE_SIMTRH, IH1FORM
      IJSM = MULTD2H(ISM,JSM)
      GETH1EX = 0.0D0
      IF(I_USE_SIMTRH.EQ.0) THEN
      IF(IH1FORM.EQ.1) THEN
*. Normal integrals, lower triangular packed
        IF(IJSM.EQ.1) THEN
C?        WRITE(6,*) ' GETH1EX, old route '
          GETH1EX =
     &    GTH1ES(IREOTS,WORK(KPINT1),H,IBSO,MXPNGAS,
     &           IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,NINOB)
        ELSE
          GETH1EX =
     &    GTH1ES(IREOTS,WORK(KPGINT1(IJSM)),H,IBSO,MXPNGAS,
     &           IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,NINOB)
        END IF
      ELSE
*. Integrals are in full blocked form
        GETH1EX =
     &  GTH1ES(IREOTS,WORK(KPGINT1A(IJSM)),H,IBSO,MXPNGAS,
     &         IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,NINOB)
      END IF
      ELSE IF (I_USE_SIMTRH.EQ.1) THEN
*. Use T1 transformed integrals
C?      WRITE(6,*) ' GETH1EX, new route '
        GETH1EX =
     &  GTH1ES(IREOTS,WORK(KPINT1_SIMTRH),H,IBSO,MXPNGAS,
     &         IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,NINOB)
      END IF
*
      RETURN
      END
      FUNCTION GETH1E(IORB,ITP,ISM,JORB,JTP,JSM)
*
* One-electron integral for active
* orbitals (IORB,ITP,ISM),(JORB,JTP,JSM)
*
* The orbital symmetries are used to obtain the
* total symmetry of the operator
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'multd2h.inc'
      INCLUDE 'intform.inc'
      INCLUDE 'oper.inc'
*
      IJSM = MULTD2H(ISM,JSM)
      GETH1E = 0.0D0
*
      IF(I_USE_SIMTRH.EQ.0) THEN
        IF(IH1FORM.EQ.1) THEN
*. Normal integrals, lower triangular packed
          IF(IJSM.EQ.1) THEN
C?        WRITE(6,*) ' GETH1E, old route '
            IF (I_UNRORB.EQ.0.OR.ISPCAS.EQ.1) THEN
              GETH1E =
     &           GTH1ES(IREOTS,WORK(KPINT1),WORK(KINT1),IBSO,MXPNGAS,
     &           IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,NINOB)
            ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
              GETH1E =
     &           GTH1ES(IREOTS,WORK(KPINT1),WORK(KINT1B),IBSO,MXPNGAS,
     &           IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,NINOB)
            ELSE
              WRITE(6,*) 'Impossible I_UNRORB/ISPCAS combination: ',
     &             I_UNRORB,ISPCAS
              STOP 'geth1e (1)'
            END IF
          ELSE
            IF (I_UNRORB.EQ.0.OR.ISPCAS.EQ.1) THEN
              GETH1E =
     &             GTH1ES(IREOTS,WORK(KPGINT1(IJSM)),WORK(KINT1),IBSO,
     &             MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,
     &             NINOB)
            ELSE  IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
              GETH1E =
     &             GTH1ES(IREOTS,WORK(KPGINT1(IJSM)),WORK(KINT1B),IBSO,
     &             MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,1,
     &             NINOB)
            ELSE
              WRITE(6,*) 'Impossible I_UNRORB/ISPCAS combination: ',
     &             I_UNRORB,ISPCAS
              STOP 'geth1e (2)'
            END IF
          END IF
        ELSE
*. Integrals are in full blocked form
          IF (I_UNRORB.EQ.0.OR.ISPCAS.EQ.1) THEN
            GETH1E =
     &      GTH1ES(IREOTS,WORK(KPGINT1A(IJSM)),WORK(KINT1),IBSO,
     &         MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,
     &         NINOB)
          ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
            GETH1E =
     &      GTH1ES(IREOTS,WORK(KPGINT1A(IJSM)),WORK(KINT1B),IBSO,
     &           MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,
     &         NINOB)
          ELSE
            WRITE(6,*) 'Impossible I_UNRORB/ISPCAS combination: ',
     &           I_UNRORB,ISPCAS
            STOP 'geth1e (3)'
          END IF
        END IF
      ELSE IF (I_USE_SIMTRH.EQ.1) THEN
*. Use T1 transformed integrals
        IF (I_UNRORB.EQ.0) THEN
          GETH1E =
     &  GTH1ES(IREOTS,WORK(KPINT1_SIMTRH),WORK(KINT1_SIMTRH),IBSO,
     &         MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,
     &         NINOB)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.1) THEN
          GETH1E =
     &  GTH1ES(IREOTS,WORK(KPINT1_SIMTRH),WORK(KINT1_SIMTRH_A),IBSO,
     &         MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,
     &         NINOB)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
          GETH1E =
     &  GTH1ES(IREOTS,WORK(KPINT1_SIMTRH),WORK(KINT1_SIMTRH_B),IBSO,
     &         MXPNGAS,IOBPTS_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,0,
     &         NINOB)
        ELSE
          WRITE(6,*) 'Impossible I_UNRORB/ISPCAS combination: ',
     &         I_UNRORB,ISPCAS
          STOP 'geth1e (4)'
        END IF
      END IF
*

      RETURN
      END
      SUBROUTINE GETINC(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,INTLST,IJKLOF,NSMOB)
*
* Obtain integrals XINT(IK,JL) = (IJ!KL) for IXCHNG = 0
*                              = (IJ!KL)-(IL!KJ) for IXCHNG = 1
*
* Version for integrals stored in INTLST
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*.LUNDIO
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
*. Integral list
      Real * 8 Intlst(*)
      Dimension IJKLof(NsmOB,NsmOb,NsmOB)
*.Output
      DIMENSION XINT(*)
*.Local
      Parameter ( lBuf    = 9600 )
COLD  Dimension Scr(lBuf)
      Logical iSymj,kSyml,iSymk,jSyml,ijSymkl,ikSymjl
      Logical ijPerm,klPerm,ijklPerm
*
C     write(6,*) ' GETINC: '
C     write(6,*) ' NSMOB ', NSMOB
C     WRITE(6,*) ' first 64 elementsin IJKLOF '
C     CALL IWRTMA(IJKLof,1,64,1,64)
      iOrb=NOBPTS(ITP,ISM)
      jOrb=NOBPTS(JTP,JSM)
      kOrb=NOBPTS(KTP,KSM)
      lOrb=NOBPTS(LTP,LSM)
*. Offsets relative to start of symmetry block 
      IOFF = 1
      DO IITP = 1, ITP -1
        IOFF = IOFF + NOBPTS(IITP,ISM)
      END DO
*
      JOFF = 1
      DO JJTP = 1, JTP -1
        JOFF = JOFF + NOBPTS(JJTP,JSM)
      END DO
*
      KOFF = 1
      DO KKTP = 1, KTP -1
        KOFF = KOFF + NOBPTS(KKTP,KSM)
      END DO
*
      LOFF = 1
      DO LLTP = 1, LTP -1
        LOFF = LOFF + NOBPTS(LLTP,LSM)
      END DO

C     iOff=IOBPTS(ITP,ISM)-IOBPTS(1,ISM)+1
C     jOff=IOBPTS(JTP,JSM)-IOBPTS(1,JSM)+1
C     kOff=IOBPTS(KTP,KSM)-IOBPTS(1,KSM)+1
C     lOff=IOBPTS(LTP,LSM)-IOBPTS(1,LSM)+1
*
*     Collect Coulmb terms
*
      ijPerm=.false.
      klPerm=.false.
      ijklPerm=.false.
      If ( iSm.gt.jSm ) Then
        iSym=iSm
        jSym=jSm
      Else
        iSym=jSm
        jSym=iSm
        ijPerm=.true.
      End If
      ijBlk=jSym+iSym*(iSym-1)/2
      If ( kSm.gt.lSm ) Then
        kSym=kSm
        lSym=lSm
      Else
        kSym=lSm
        lSym=kSm
        klPerm=.true.
      End If
      klBlk=lSym+kSym*(kSym-1)/2
      If ( klBlk.gt.ijBlk ) Then
        iTemp=iSym
        iSym=kSym
        kSym=iTemp
        iTemp=jSym
        jSym=lSym
        lSym=iTemp
        iTemp=ijBlk
        ijBlk=klBlk
        klBlk=iTemp
        ijklPerm=.true.
      End If
C?    print *,' GetInt reports to you'
C?    print *,' iSym,jSym,kSym,lSym',iSym,jSym,kSym,lSym
      iSyBlk=klBlk+ijBlk*(ijBlk-1)/2
      iBatch=nBatch(iSyBlk)
CJO Define offset for given symmetry block
      IBLoff = IJKLof(Isym,Jsym,Ksym)
C?    WRITE(6,*) ' IBLoff Isym Jsym Ksym ', IBLoff,ISym,Jsym,Ksym
C?    print *,' iSyBlk,iBatch',iSyBlk,iBatch
      iSymj=iSym.eq.jSym
      kSyml=kSym.eq.lSym
      iSymk=iSym.eq.kSym
      jSyml=jSym.eq.lSym
      ijSymkl=iSymj.and.kSyml
      ikSymjl=iSymk.and.jSyml
      itOrb=NTOOBS(iSym)
      jtOrb=NTOOBS(jSym)
      ktOrb=NTOOBS(kSym)
      ltOrb=NTOOBS(lSym)
C?    print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
      ijPairs=itOrb*jtOrb
      klPairs=ktOrb*ltOrb
      nInts=ijPairs*klPairs
      If ( ikSymjl ) Then
        If ( iSymj ) Then
          ijPairs=itOrb*(itOrb+1)/2
          klPairs=ijPairs
          nInts=ijPairs*(klPairs+1)/2
        Else
          ijPairs=itOrb*jtOrb
          klPairs=ijPairs
          nInts=ijPairs*(klPairs+1)/2
        End If
      Else If ( ijSymkl ) Then
        ijPairs=itOrb*(itOrb+1)/2
        klPairs=ktOrb*(ktOrb+1)/2
        nInts=ijPairs*klPairs
      End If
C?    print *,' ijPairs,klPairs',ijPairs,klPairs
      iInt=0
C?    print *,' Start loop over Coulomb integrals'
C?    print *,' ijkl,ij,kl,i,j,k,l'
      Do lJeppe=lOff,lOff+lOrb-1
        jMin=jOff
        If ( JLSM.ne.0 ) jMin=lJeppe
        Do jJeppe=jMin,jOff+jOrb-1
          Do kJeppe=kOff,kOff+kOrb-1
            iMin = iOff
            If(IKSM.ne.0) iMin = kJeppe
            Do iJeppe=iMin,iOff+iOrb-1
              If ( ijPerm ) Then
                ii=jJeppe
                jj=iJeppe
              Else
                ii=iJeppe
                jj=jJeppe
              End If
              If ( klPerm ) Then
                kk=lJeppe
                ll=kJeppe
              Else
                kk=kJeppe
                ll=lJeppe
              End If
              If ( ijklPerm ) Then
                i=kk
                j=ll
                k=ii
                l=jj
              Else
                i=ii
                j=jj
                k=kk
                l=ll
              End If
              If ( iSymj ) Then
                If ( i.gt.j ) Then
                  ij=j+i*(i-1)/2
                Else
                  ij=i+j*(j-1)/2
                End If
              Else
                ij=j+(i-1)*jtOrb
              End If
              ijOff=ij+(ij-1)*(ij-2)/2-1
              If ( kSyml ) Then
                If ( k.gt.l ) Then
                  kl=l+k*(k-1)/2
                Else
                  kl=k+l*(l-1)/2
                End If
              Else
                kl=l+(k-1)*ltOrb
              End If
              klOff=kl+(kl-1)*(kl-2)/2-1
              If ( ikSymjl ) Then
                If ( ij.gt.kl ) Then
                  ijkl=ij+(kl-1)*ijPairs-klOff
                Else
                  ijkl=kl+(ij-1)*klPairs-ijOff
                End If
              Else
                ijkl=ij+(kl-1)*ijPairs
              End If
C?            print '(7i6)',ijkl,ij,kl,i,j,k,l
              iInt=iInt+1
              Xint(iInt) = Intlst(iblOff-1+ijkl)
            End Do
          End Do
        End Do
      End Do
*
*     Collect Exchange terms
*
      If ( IXCHNG.ne.0 ) Then
        ijPerm=.false.
        klPerm=.false.
        ijklPerm=.false.
        If ( iSm.gt.lSm ) Then
          iSym=iSm
          jSym=lSm
        Else
          iSym=lSm
          jSym=iSm
          ijPerm =.True.
        End If
        ijBlk=jSym+iSym*(iSym-1)/2
        If ( kSm.gt.jSm ) Then
          kSym=kSm
          lSym=jSm
        Else
          kSym=jSm
          lSym=kSm
          klPerm =.True.
        End If
        klBlk=lSym+kSym*(kSym-1)/2
        If ( klBlk.gt.ijBlk ) Then
          iTemp=iSym
          iSym=kSym
          kSym=iTemp
          iTemp=jSym
          jSym=lSym
          lSym=iTemp
          iTemp=ijBlk
          ijBlk=klBlk
          klBlk=iTemp
          ijklPerm = .True.
        End If
        iSyBlk=klBlk+ijBlk*(ijBlk-1)/2
*
        Ibloff = IJKLof(Isym,Jsym,Ksym)
        iSymj=iSym.eq.jSym
        kSyml=kSym.eq.lSym
        iSymk=iSym.eq.kSym
        jSyml=jSym.eq.lSym
        ijSymkl=iSymj.and.kSyml
        ikSymjl=iSymk.and.jSyml
        itOrb=NTOOBS(iSym)
        jtOrb=NTOOBS(jSym)
        ktOrb=NTOOBS(kSym)
        ltOrb=NTOOBS(lSym)
        ijPairs=itOrb*jtOrb
        klPairs=ktOrb*ltOrb
        nInts=ijPairs*klPairs
        If ( ikSymjl ) Then
          If ( iSymj ) Then
            ijPairs=itOrb*(itOrb+1)/2
            klPairs=ijPairs
            nInts=ijPairs*(klPairs+1)/2
          Else
            ijPairs=itOrb*jtOrb
            klPairs=ijPairs
            nInts=ijPairs*(klPairs+1)/2
          End If
        Else If ( ijSymkl ) Then
          ijPairs=itOrb*(itOrb+1)/2
          klPairs=ktOrb*(ktOrb+1)/2
          nInts=ijPairs*klPairs
        End If
        iInt=0
C?      print *,' iSym,jSym,kSym,lSym',iSym,jSym,kSym,lSym
C?      print *,' iSyBlk,iBatch',iSyBlk,iBatch
C?      print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
C?      print *,' ijPairs,klPairs',ijPairs,klPairs
C?      print *,' Start loop over Exchange integrals'
C?      print *,' ijkl,ij,kl,i,j,k,l'
        Do lJeppe=lOff,lOff+lOrb-1
          jMin=jOff
          If ( JLSM.ne.0 ) jMin=lJeppe
          Do jJeppe=jMin,jOff+jOrb-1
            Do kJeppe=kOff,kOff+kOrb-1
              iMin = iOff
              If(IKSM.ne.0) iMin = kJeppe
              Do iJeppe=iMin,iOff+iOrb-1
              If ( ijPerm ) Then
                ii=lJeppe
                ll=iJeppe
              Else
                ii=iJeppe
                ll=lJeppe
              End If
              If ( klPerm ) Then
                kk=jJeppe
                jj=kJeppe
              Else
                kk=kJeppe
                jj=jJeppe
              End If
              If ( ijklPerm ) Then
                i=kk
                l=jj
                k=ii
                j=ll
              Else
                i=ii
                j=jj
                k=kk
                l=ll
              End If
                If ( iSymj ) Then
                  If ( i.gt.l ) Then
                    ij=l+i*(i-1)/2
                  Else
                    ij=i+l*(l-1)/2
                  End If
                Else
                  ij=l+(i-1)*jtOrb
                End If
                ijOff=ij+(ij-1)*(ij-2)/2-1
                If ( kSyml ) Then
                  If ( k.gt.j ) Then
                    kl=j+k*(k-1)/2
                  Else
                    kl=k+j*(j-1)/2
                  End If
                Else
                  kl=j+(k-1)*ltOrb
                End If
                klOff=kl+(kl-1)*(kl-2)/2-1
                If ( ikSymjl ) Then
                  If ( ij.gt.kl ) Then
                    ijkl=ij+(kl-1)*ijPairs-klOff
                  Else
                    ijkl=kl+(ij-1)*klPairs-ijOff
                  End If
                Else
                  ijkl=ij+(kl-1)*ijPairs
                End If
C?              print '(7i6)',ijkl,ij,kl,i,j,k,l
                iInt=iInt+1
                XInt(iInt)=XInt(iInt)-Intlst(iBLoff-1+ijkl)
              End Do
            End Do
          End Do
        End Do
      End If
*
      Return
      End
      SUBROUTINE GETINL(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM)
*
* Obtain integrals XINT(IK,JL) = (IJ!KL) for IXCHNG = 0
*                              = (IJ!KL)-(IL!KJ) for IXCHNG = 1
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*.Output
      DIMENSION XINT(*)
*
      CALL QENTER('GETIN')
      NI = NTSOB(ITP,ISM)
      NJ = NTSOB(JTP,JSM)
      NK = NTSOB(KTP,KSM)
      NL = NTSOB(LTP,LSM)
*
      IF(IKSM.NE.0) THEN
        NIK = NI*(NI+1)/2
      ELSE
        NIK = NI*NK
      END IF
      IF(JLSM.NE.0) THEN
        NJL = NJ*(NJ+1)/2
      ELSE
        NJL = NJ*NL
      END IF
*
      I = 0
      K = 1
      DO 100 IK = 1, NIK
C       CALL NXTIJ(L,J,NL,NJ,JLSM,NONEW)
        CALL NXTIJ(I,K,NI,NK,IKSM,NONIK)
        IABS = I + IBTSOB(ITP,ISM)-1
        KABS = K + IBTSOB(KTP,KSM)-1
        J = 0
        L = 1
        DO 50 JL = 1, NJL
          CALL NXTIJ(J,L,NJ,NL,JLSM,NONJL)
          LABS = L + IBTSOB(LTP,LSM)-1
          JABS = J + IBTSOB(JTP,JSM)-1
C         XINT((IK-1)*NJL+JL) = GTIJKL(IABS,JABS,KABS,LABS)
C         IF(IXCHNG.EQ.1)
C    &    XINT((IK-1)*NJL+JL) = XINT((IK-1)*NJL+JL)
C    &    - GTIJKL(IABS,LABS,KABS,JABS)
          XINT((JL-1)*NIK+IK) = GTIJKL(IABS,JABS,KABS,LABS)
          IF(IXCHNG.EQ.1)
     &    XINT((JL-1)*NIK+IK) = XINT((JL-1)*NIK+IK)
     &    - GTIJKL(IABS,LABS,KABS,JABS)
   50   CONTINUE
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' 2 electron integral block for TS blocks '
        WRITE(6,'(1H ,4(A,I2,A,I2,A))')
     &  '(',ITP,',',ISM,')','(',JTP,',',JSM,')',
     &  '(',KTP,',',KSM,')','(',LTP,',',LSM,')'
        CALL WRTMAT(XINT,NIK,NJL,NIK,NJL)
      END IF
*
      CALL QEXIT('GETIN')
      RETURN
      END
      SUBROUTINE GETINM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM)
*
* Obtain integrals XINT(IK,JL) = (IJ!KL) for IXCHNG = 0
*                              = (IJ!KL)-(IL!KJ) for IXCHNG = 1
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*.LUNDIO
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
*.Output
      DIMENSION XINT(*)
*.Local
      Parameter ( lBuf    = 9600 )
      Dimension Scr(lBuf)
      Logical iSymj,kSyml,iSymk,jSyml,ijSymkl,ikSymjl
      Logical ijPerm,klPerm,ijklPerm
*
      iOrb=NOBPTS(ITP,ISM)
      jOrb=NOBPTS(JTP,JSM)
      kOrb=NOBPTS(KTP,KSM)
      lOrb=NOBPTS(LTP,LSM)
      iOff=IOBPTS(ITP,ISM)
      jOff=IOBPTS(JTP,JSM)
      kOff=IOBPTS(KTP,KSM)
      lOff=IOBPTS(LTP,LSM)
*
*     Collect Coulmb terms
*
      ijPerm=.false.
      klPerm=.false.
      ijklPerm=.false.
      If ( iSm.gt.jSm ) Then
        iSym=iSm
        jSym=jSm
      Else
        iSym=jSm
        jSym=iSm
        ijPerm=.true.
      End If
      ijBlk=jSym+iSym*(iSym-1)/2
      If ( kSm.gt.lSm ) Then
        kSym=kSm
        lSym=lSm
      Else
        kSym=lSm
        lSym=kSm
        klPerm=.true.
      End If
      klBlk=lSym+kSym*(kSym-1)/2
      If ( klBlk.gt.ijBlk ) Then
        iTemp=iSym
        iSym=kSym
        kSym=iTemp
        iTemp=jSym
        jSym=lSym
        lSym=iTemp
        iTemp=ijBlk
        ijBlk=klBlk
        klBlk=iTemp
        ijklPerm=.true.
      End If
C?    print *,' GetInt reports to you'
C?    print *,' iSym,jSym,kSym,lSym',iSym,jSym,kSym,lSym
      iSyBlk=klBlk+ijBlk*(ijBlk-1)/2
      iBatch=nBatch(iSyBlk)
C?    print *,' iSyBlk,iBatch',iSyBlk,iBatch
      iSymj=iSym.eq.jSym
      kSyml=kSym.eq.lSym
      iSymk=iSym.eq.kSym
      jSyml=jSym.eq.lSym
      ijSymkl=iSymj.and.kSyml
      ikSymjl=iSymk.and.jSyml
      itOrb=NTOOBS(iSym)
      jtOrb=NTOOBS(jSym)
      ktOrb=NTOOBS(kSym)
      ltOrb=NTOOBS(lSym)
C?    print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
      ijPairs=itOrb*jtOrb
      klPairs=ktOrb*ltOrb
      nInts=ijPairs*klPairs
      If ( ikSymjl ) Then
        If ( iSymj ) Then
          ijPairs=itOrb*(itOrb+1)/2
          klPairs=ijPairs
          nInts=ijPairs*(klPairs+1)/2
        Else
          ijPairs=itOrb*jtOrb
          klPairs=ijPairs
          nInts=ijPairs*(klPairs+1)/2
        End If
      Else If ( ijSymkl ) Then
        ijPairs=itOrb*(itOrb+1)/2
        klPairs=ktOrb*(ktOrb+1)/2
        nInts=ijPairs*klPairs
      End If
C?    print *,' ijPairs,klPairs',ijPairs,klPairs
      iRecOld=-1
      iInt=0
C?    print *,' Start loop over Coulomb integrals'
C?    print *,' ijkl,ij,kl,i,j,k,l'
      Do lJeppe=lOff,lOff+lOrb-1
        jMin=jOff
        If ( JLSM.ne.0 ) jMin=lJeppe
        Do jJeppe=jMin,jOff+jOrb-1
          Do kJeppe=kOff,kOff+kOrb-1
            iMin = iOff
            If(IKSM.ne.0) iMin = kJeppe
            Do iJeppe=iMin,iOff+iOrb-1
              If ( ijPerm ) Then
                ii=jJeppe
                jj=iJeppe
              Else
                ii=iJeppe
                jj=jJeppe
              End If
              If ( klPerm ) Then
                kk=lJeppe
                ll=kJeppe
              Else
                kk=kJeppe
                ll=lJeppe
              End If
              If ( ijklPerm ) Then
                i=kk
                j=ll
                k=ii
                l=jj
              Else
                i=ii
                j=jj
                k=kk
                l=ll
              End If
              If ( iSymj ) Then
                If ( i.gt.j ) Then
                  ij=j+i*(i-1)/2
                Else
                  ij=i+j*(j-1)/2
                End If
              Else
                ij=j+(i-1)*jtOrb
              End If
              ijOff=ij+(ij-1)*(ij-2)/2-1
              If ( kSyml ) Then
                If ( k.gt.l ) Then
                  kl=l+k*(k-1)/2
                Else
                  kl=k+l*(l-1)/2
                End If
              Else
                kl=l+(k-1)*ltOrb
              End If
              klOff=kl+(kl-1)*(kl-2)/2-1
              If ( ikSymjl ) Then
                If ( ij.gt.kl ) Then
                  ijkl=ij+(kl-1)*ijPairs-klOff
                Else
                  ijkl=kl+(ij-1)*klPairs-ijOff
                End If
              Else
                ijkl=ij+(kl-1)*ijPairs
              End If
C?            print '(7i6)',ijkl,ij,kl,i,j,k,l
              iRec=(ijkl-1)/lTr2Rec
              If ( iRec.eq.iRecOld ) then
                ijkl=ijkl-iRec*lTr2Rec
              Else
                iDisk=iDAdr(iBatch)
                Do iSkip=1,iRec
                  Call DaFile(LuTr2,0,Scr,2*lTr2Rec,iDisk)
                End Do
                Call DaFile(LuTr2,2,Scr,2*lTr2Rec,iDisk)
                ijkl=ijkl-iRec*lTr2Rec
                iRecOld=iRec
              End If
              iInt=iInt+1
              XInt(iInt)=Scr(ijkl)
            End Do
          End Do
        End Do
      End Do
*
*     Collect Exchange terms
*
      If ( IXCHNG.ne.0 ) Then
        ijPerm=.false.
        klPerm=.false.
        ijklPerm=.false.
        If ( iSm.gt.lSm ) Then
          iSym=iSm
          jSym=lSm
        Else
          iSym=lSm
          jSym=iSm
          ijPerm =.True.
        End If
        ijBlk=jSym+iSym*(iSym-1)/2
        If ( kSm.gt.jSm ) Then
          kSym=kSm
          lSym=jSm
        Else
          kSym=jSm
          lSym=kSm
          klPerm =.True.
        End If
        klBlk=lSym+kSym*(kSym-1)/2
        If ( klBlk.gt.ijBlk ) Then
          iTemp=iSym
          iSym=kSym
          kSym=iTemp
          iTemp=jSym
          jSym=lSym
          lSym=iTemp
          iTemp=ijBlk
          ijBlk=klBlk
          klBlk=iTemp
          ijklPerm = .True.
        End If
        iSyBlk=klBlk+ijBlk*(ijBlk-1)/2
        iBatch=nBatch(iSyBlk)
        iSymj=iSym.eq.jSym
        kSyml=kSym.eq.lSym
        iSymk=iSym.eq.kSym
        jSyml=jSym.eq.lSym
        ijSymkl=iSymj.and.kSyml
        ikSymjl=iSymk.and.jSyml
        itOrb=NTOOBS(iSym)
        jtOrb=NTOOBS(jSym)
        ktOrb=NTOOBS(kSym)
        ltOrb=NTOOBS(lSym)
        ijPairs=itOrb*jtOrb
        klPairs=ktOrb*ltOrb
        nInts=ijPairs*klPairs
        If ( ikSymjl ) Then
          If ( iSymj ) Then
            ijPairs=itOrb*(itOrb+1)/2
            klPairs=ijPairs
            nInts=ijPairs*(klPairs+1)/2
          Else
            ijPairs=itOrb*jtOrb
            klPairs=ijPairs
            nInts=ijPairs*(klPairs+1)/2
          End If
        Else If ( ijSymkl ) Then
          ijPairs=itOrb*(itOrb+1)/2
          klPairs=ktOrb*(ktOrb+1)/2
          nInts=ijPairs*klPairs
        End If
        iRecOld=-1
        iInt=0
C?      print *,' iSym,jSym,kSym,lSym',iSym,jSym,kSym,lSym
C?      print *,' iSyBlk,iBatch',iSyBlk,iBatch
C?      print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
C?      print *,' ijPairs,klPairs',ijPairs,klPairs
C?      print *,' Start loop over Exchange integrals'
C?      print *,' ijkl,ij,kl,i,j,k,l'
        Do lJeppe=lOff,lOff+lOrb-1
          jMin=jOff
          If ( JLSM.ne.0 ) jMin=lJeppe
          Do jJeppe=jMin,jOff+jOrb-1
            Do kJeppe=kOff,kOff+kOrb-1
              iMin = iOff
              If(IKSM.ne.0) iMin = kJeppe
              Do iJeppe=iMin,iOff+iOrb-1
              If ( ijPerm ) Then
                ii=lJeppe
                ll=iJeppe
              Else
                ii=iJeppe
                ll=lJeppe
              End If
              If ( klPerm ) Then
                kk=jJeppe
                jj=kJeppe
              Else
                kk=kJeppe
                jj=jJeppe
              End If
              If ( ijklPerm ) Then
                i=kk
                l=jj
                k=ii
                j=ll
              Else
                i=ii
                j=jj
                k=kk
                l=ll
              End If
                If ( iSymj ) Then
                  If ( i.gt.l ) Then
                    ij=l+i*(i-1)/2
                  Else
                    ij=i+l*(l-1)/2
                  End If
                Else
                  ij=l+(i-1)*jtOrb
                End If
                ijOff=ij+(ij-1)*(ij-2)/2-1
                If ( kSyml ) Then
                  If ( k.gt.j ) Then
                    kl=j+k*(k-1)/2
                  Else
                    kl=k+j*(j-1)/2
                  End If
                Else
                  kl=j+(k-1)*ltOrb
                End If
                klOff=kl+(kl-1)*(kl-2)/2-1
                If ( ikSymjl ) Then
                  If ( ij.gt.kl ) Then
                    ijkl=ij+(kl-1)*ijPairs-klOff
                  Else
                    ijkl=kl+(ij-1)*klPairs-ijOff
                  End If
                Else
                  ijkl=ij+(kl-1)*ijPairs
                End If
C?              print '(7i6)',ijkl,ij,kl,i,j,k,l
                iRec=(ijkl-1)/lTr2Rec
                If ( iRec.eq.iRecOld ) then
                  ijkl=ijkl-iRec*lTr2Rec
                Else
                  iDisk=iDAdr(iBatch)
                  Do iSkip=1,iRec
                    Call DaFile(LuTr2,0,Scr,2*lTr2Rec,iDisk)
                  End Do
                  Call DaFile(LuTr2,2,Scr,2*lTr2Rec,iDisk)
                  ijkl=ijkl-iRec*lTr2Rec
                  iRecOld=iRec
                End If
                iInt=iInt+1
                XInt(iInt)=XInt(iInt)-Scr(ijkl)
              End Do
            End Do
          End Do
        End Do
      End If
*
      Return
      End
      SUBROUTINE GETINT(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,ICOUL,CFACX,EFACX)
*
*
* Outer routine for accessing integral block
*
* if we have unrestricted spin-orbitals (I_UNRORB.EQ.1), this is important:
* ISPCAS: 1 -- alpha alpha
* ISPCAS: 2 -- beta  beta
* ISPCAS: 3 -- alpha beta (i.e. IJ alpha, KL beta)
* ISPCAS: 4 -- beta  alpha(i.e. IJ beta,  KL alpha)
*
c      IMPLICIT REAL*8(A-H,O-Z)
*
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cgas.inc'
*. Local scratch: 
*
      CALL QENTER('GETIN')
      NTEST = 00
C     IF(ISM.EQ.2.AND.JSM.EQ.5.AND.KSM.EQ.1.AND.LSM.EQ.6) THEN
C        NTEST = 100
C        WRITE(6,*) ' Jeppe raised print level in GETINT'
C     END IF
*
      CFAC = CFACX
      EFAC = EFACX
*
*
      IF(NTEST.GE.1) THEN
       WRITE(6,*) ' Information from GETINT '
       WRITE(6,*) ' ======================='
       WRITE(6,*) ' I_USE_SIMTRH in GETINT =', I_USE_SIMTRH
       WRITE(6,*) ' I_UNRORB in GETINT =', I_UNRORB
       WRITE(6,*) ' GETINT: ICC_EXC and ICOUL = ', ICC_EXC, ICOUL
       WRITE(6,*)       'ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM: ' 
       WRITE(6,'(8I4)')  ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM
       WRITE(6,'(A,I2)') ' ITRA_ROUTE = ', ITRA_ROUTE

      END IF
*
*. Modified July 2010: ITP = -1 now indicated all orbitals
*                          =  0 indicates inactive
*                          =  NGAS + 1 indicates secondary
*  ITRA_ROUTE added May 2011, Jeppe Olsen
*
      IF (ISPCAS.EQ.4) STOP 'STILL A BUG FOR ISPCAS.EQ.4!'
*
      IF(ITRA_ROUTE.EQ.1) THEN
*
* ==========================
* Standard (pre 2011 route)
* ==========================
*
       IF(ICC_EXC.EQ.0.AND.I_USE_SIMTRH.EQ.0) THEN
*
* =======================
* Usual/Normal  integrals
* =======================
*
*. Integrals in core in internal LUCIA format
        IF(ICOUL.NE.2.AND.I_UNRORB.EQ.0) THEN
          IF (I12S.EQ.1.AND.I34S.EQ.1) THEN
            CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                    IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                    WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0,CFAC,
     &                    EFAC)
          ELSE
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
          END IF
        ELSE IF (I_UNRORB.EQ.0) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0,CFAC,
     &                    EFAC)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.1) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0,CFAC,
     &                    EFAC)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2BB),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0,CFAC,
     &                    EFAC)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.3) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2AB),
     &                  WORK(KPINT2AB),NSMOB,WORK(KINH1),ICOUL,1,CFAC,
     &                    EFAC)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.4) THEN
          CALL GETINCN2(XINT,KTP,KSM,LTP,LSM,ITP,ISM,JTP,JSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2AB),
     &                  WORK(KPINT2AB),NSMOB,WORK(KINH1),ICOUL,1,CFAC,
     &                    EFAC)
        ELSE
          WRITE(6,*) 'WRONG ISPCAS IN GETINT (',ISPCAS,')'
          STOP 'getint'
        END IF
       ELSE IF (ICC_EXC.EQ.1.AND.I_USE_SIMTRH.EQ.0) THEN
*
* ============================
* Coupled Cluster coefficients 
* ============================
* 
        IF(ICOUL.EQ.1) THEN
          IKLJ = 0 
          IJ_TRNSP = 1
        ELSE
          IKLJ = 1
          IJ_TRNSP = 0
        END IF
*. IJ_TRNSP: RSBB2BN requires blocks for e(ijkl) in the form C(ji,kl)
*. Amplitudes fetched from KCC1, KCC2 used as scratch 
        CALL GET_DX_BLK(ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,WORK(KCC1+NSXE),
     &                  XINT,1,IXCHNG,IKLJ,IKSM,JLSM,WORK(KCC2),
     &                  IJ_TRNSP )
C            GET_DX_BLK(IGAS,ISM,JGAS,JSM,KGAS,KSM,LGAS,LSM, 
C    &                  C,CBLK,IEXP,IXCHNG,IKLJ,IKSM,JLSM,SCR)
       ELSE IF( I_USE_SIMTRH.EQ.1) THEN
*. Use similarity transformed integrals
        IF(I_UNRORB.EQ.0) THEN
C          IF(ICOUL.NE.2) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0,CFAC,
     &                    EFAC)
C          ELSE
C            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
C     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH),
C     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
C     &                  ICOUL,0)
C          END IF

        ELSE
          IF(ISPCAS.EQ.1) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AA),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0)
          ELSE IF(ISPCAS.EQ.2) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_BB),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0)
          ELSE IF(ISPCAS.EQ.3) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                 IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AB),
     &                 WORK(KPINT2_SIMTRH_AB),NSMOB,WORK(KINH1_NOCCSYM),
     &                 ICOUL,1)
          ELSE IF(ISPCAS.EQ.4) THEN
            CALL GETINCN2_NOCCSYM(XINT,KTP,KSM,LTP,LSM,ITP,ISM,JTP,JSM,
     &                 IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AB),
     &                 WORK(KPINT2_SIMTRH_AB),NSMOB,WORK(KINH1_NOCCSYM),
     &                 ICOUL,1)
          ELSE
            WRITE(6,*) 'WRONG ISPCAS IN GETINT (',ISPCAS,')'
            STOP 'getint'
          END IF
        END IF
       END IF
      ELSE IF (ITRA_ROUTE.EQ.2) THEN
*. General transformation. 
*. Determine first type of integral, that ITP,JTP,KTP, LTP belongs to
*
* ==========================
* New (2011 route)
* ==========================
*
*. Determine (first) active array containing this integral block
        CALL GET_INTARR_F4TP(INTARR,ITP,JTP,KTP,LTP)
*. Permutational symmetry of this block
        I12S_A = I12S_G(INTARR)
        I34S_A = I34S_G(INTARR)
        I1234S_A = I1234S_G(INTARR)
        IOCOBTP_A = IOCOBTP_G(INTARR)
        KINT2_LA = KINT2_A(INTARR)
        KPINT2_LA = KPINT2_A(INTARR)
        IF(NTEST.GE.100) THEN
          WRITE(6,'(A,5I3)') 
     &    ' INTARR, I12S_A, I34S_A, I1234S_A, IOCOBTP_A = ',
     &      INTARR, I12S_A, I34S_A, I1234S_A, IOCOBTP_A
          WRITE(6,'(A,2I9)')  
     &    ' KINT2_LA, KPINT2_LA = ', KINT2_LA, KPINT2_LA
        END IF
*. Set up orbital arrays for this integral array
        IDIM_ONLY = 1
        XDUM = -123456789.0D0
        CALL GET_DIM_AND_C_FOR_ORBS(IOCOBTP_A,INT2ARR_G(1,INTARR),
     &       NTOOBS_IA,NTOOBS_JA,NTOOBS_KA,NTOOBS_LA,
     &       NOBPTS_GN_A(0,1,1),NOBPTS_GN_A(0,1,2),
     &       NOBPTS_GN_A(0,1,3),NOBPTS_GN_A(0,1,4),
     &       XDUM,XDUM,XDUM,XDUM,
     &       XDUM,XDUM,XDUM,XDUM,IDIM_ONLY) 
*. And then the integral block
C       GETINCN2_A(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
C    &             IXCHNG,IKSM,JLSM,INTLST,IJKLOF,I2INDX,
C    &             ICOUL,CFAC,EFAC)
         CALL GETINCN2_A(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                    IXCHNG,IKSM,JLSM,WORK(KINT2_LA),
     &                    WORK(KPINT2_LA),WORK(KINH1),ICOUL,CFAC,
     &                    EFAC)
      END IF! End of ITRA_ROUTE switch
*
      IF(NTEST.GE.100) THEN
        IF(ITP.EQ.-1) THEN
          NI = NTOOBS(ISM)
        ELSE
          NI = NOBPTS_GN(ITP,ISM)
        END IF
        IF(KTP.EQ.-1) THEN
          NK = NTOOBS(KSM)
        ELSE
          NK = NOBPTS_GN(KTP,KSM)
        END IF
*
        IF(IKSM.EQ.0) THEN
          NIK = NI * NK
        ELSE
          NIK = NI*(NI+1)/2
        END IF
*
        IF(JTP.EQ.-1) THEN
         NJ = NTOOBS(JSM)
        ELSE
         NJ = NOBPTS_GN(JTP,JSM)
        END IF
        IF(LTP.EQ.-1) THEN
          NL = NTOOBS(LSM)
        ELSE
          NL = NOBPTS_GN(LTP,LSM)
        END IF
*
        IF(JLSM.EQ.0) THEN
          NJL = NJ * NL
        ELSE
          NJL = NJ*(NJ+1)/2
        END IF
        WRITE(6,*) ' 2 electron integral block for TS blocks '
        WRITE(6,*) ' Icoul:', ICOUL
        WRITE(6,*) ' Ixchng:', IXCHNG
        WRITE(6,*) ' ISPCAS:', ISPCAS
        WRITE(6,*) ' CFAC, EFAC: ', CFAC, EFAC
        WRITE(6,'(A,4I4)') 
     &  ' NI, NJ, NK, NL = ', NI, NJ, NK, NL
        WRITE(6,*) ' Integrals from GETINT:'
        IF(ICOUL.EQ.0) THEN
          WRITE(6,'(1H ,4(A,I2,A,I2,A))')
     &    '(',ITP,',',ISM,')','(',KTP,',',KSM,')',
     &    '(',JTP,',',JSM,')','(',LTP,',',LSM,')'
          CALL WRTMAT(XINT,NIK,NJL,NIK,NJL)
        ELSE
          WRITE(6,'(1H ,4(A,I2,A,I2,A))')
     &    '(',ITP,',',ISM,')','(',JTP,',',JSM,')',
     &    '(',KTP,',',KSM,')','(',LTP,',',LSM,')'
          CALL WRTMAT(XINT,NI*NJ,NK*NL,NI*NJ,NK*NL)
        END IF
      END IF
*
      CALL QEXIT('GETIN')
COLD  STOP ' Jeppe forced me to stop in GETINT '
      RETURN
      END
      Subroutine Getobs2(ECORE,NAOS_ENV,NMOS_ENV)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Open one-electron integral file
*.    Read in orbital information  and core energy
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Implicit Real*8 (A-H,O-Z)
*
      Parameter( LuOne = 14)
*
      DIMENSION NAOS_ENV(*),NMOS_ENV(*)
*
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),Nsym
      INTEGER*8 Itoc, nBas, nOrb, nFro, nDel, Nsym

*GLM      Character*4 Name(2,400)
      Character*10 Name(800)
*----------------------------------------------------------------------*
*     Start procedure:                                                 *
*     open the transformed one-electron integral file                  *
*----------------------------------------------------------------------*
      NTEST = 10

      Call DaName2(LuOne,'TRAONE')
      iDisk=0
      Do I=1,8
        Nbas(I)=0
        Norb(I)=0
      End Do 
      CALL WR_MOTRA_Info(LuOne,2,iDisk,
     &                   iToc,64, ECore,
     &                   nSym, nBas, nOrb,nFro,nDel,8,Name,800)
      IF (NTEST.ge.10) then
       write(6,*) '*******************************************'
       write(6,*) ' >>>> In Output from WR_MOTRA_Info: <<<<'
       write(6,*) ' ECore= ', ECore
       write(6,*) ' nSym = ', nSym
       write(6,'(A9,8I3)') ' nBas = ', (nBas(i),i=1,nSym)
       write(6,'(A9,8I3)') ' nOrb = ',(nOrb(i),i=1,nSym)
       write(6,'(A9,8I3)') ' nFro = ',(nFro(i),i=1,nSym)
       write(6,'(A9,8I3)') ' nDel = ',(nDel(i),i=1,nSym)
       write(6,*) ' iToc(64)'
       write(6,'(10I5)')  iToc
       write(6,*) '*******************************************'
      End If  
*GLM      Call DaName(LuOne,'TRAONE')
*----------------------------------------------------------------------*
*     Set up the scatter/gather list and read table of contents        *
*----------------------------------------------------------------------*
*GLM      iDisk=0
*GLM      Call GSList(IOList,8,iToc,64,ECORE,2,nSym,1,
*GLM     &            nBas,8,nOrb,8,nFro,8,nDel,8,Name,800)
*GLM      Call DaFile(LuOne,4,IOList,iDum,iDisk)
*----------------------------------------------------------------------*
*     Terminate procedure                                              *
*----------------------------------------------------------------------*
*
* Copy to external arrays
*
      Do iSym = 1, nSym
        NMOS_ENV(iSym) = NORB(iSym)
        NAOS_ENV(iSym) = NBAS(iSym)
      End Do

*GLM      CALL ICOPVE(NORB,NMOS_ENV,NSYM)
*GLM      CALL ICOPVE(NBAS,NAOS_ENV,NSYM)
*
      Return
      End
      Subroutine Getobs(ECORE)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Open one-electron integral file
*.    Read in orbital information  and core energy
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Implicit Real*8 (A-H,O-Z)
*
      Parameter( LuOne = 14)
*
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
      Character*4 Name(2,400)
*----------------------------------------------------------------------*
*     Start procedure:                                                 *
*     open the transformed one-electron integral file                  *
*----------------------------------------------------------------------*
      Call DaName(LuOne,'TRAONE')
*----------------------------------------------------------------------*
*     Set up the scatter/gather list and read table of contents        *
*----------------------------------------------------------------------*
      iDisk=0
      Call GSList(IOList,8,iToc,64,ECORE,2,nSym,1,
     &            nBas,8,nOrb,8,nFro,8,nDel,8,Name,800)
      Call DaFile(LuOne,4,IOList,iDum,iDisk)
*----------------------------------------------------------------------*
*     Terminate procedure                                              *
*----------------------------------------------------------------------*
      Return
      End
      Subroutine GetobsS(ECORE)
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*.    Read in orbital information  and core energy in SIRIUS format    *
*     i.e generate /MOLOBS/ and obtain ECORE                           *
*                                                                      *
***** H. Koch + J. Olsen, A dark february evening in 1993  *************
*
      Implicit Real*8 (A-H,O-Z)
*
      Parameter( LuOne = 14)
      Parameter( ITAP  = 16)
C
      Dimension nlamda(8), nocc(8)
C
*
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
* Required information: Nbas, Norb, Nfro, nDel, Nsym
C
      NTEST = 00
      IF ( NTEST .GE. 10 ) THEN
	 WRITE(6,*) ' GETOBSS entered '
      END IF
*       
      OPEN(ITAP,STATUS='OLD',FORM='UNFORMATTED',FILE='SIRIFC')
      REWIND ITAP                                          
      CALL MOLLAB('TRCCINT ',ITAP,6)                                  
      READ (ITAP) NSYMHF,NORBT,NBAST,NCMOT,(NOCC(I),I=1,NSYMHF),      
     *            (NLAMDA(I),I=1,NSYMHF),(NORB(I),I=1,NSYMHF),   
     *            POTNUC,EMCSCF                                  
C
      Nsym = nsymhf
      IF( NORBT.NE.NBAST) THEN
        WRITE(6,*) 
     &  ' Error: Number of MO''s differs from number of AO''s'
        WRITE(6,*) ' Do not delete MO''s or reprogram me '
        WRITE(6,*) ' Untill then: I stop '
        STOP'GETOBSS: NORBT .NE. NBAST '
      END IF
      CALL ISETVC(NDEL,0,8)
      CALL ISETVC(NFRO,0,8)
      CALL ICOPVE(NORB,NBAS,8)
*. No frozen orbitals so
      ECORE = POTNUC
*
      IF( NTEST .GE. 10 ) THEN
        WRITE(6,*) ' ==================='
        WRITE(6,*) ' Output from GETOBSS'
        WRITE(6,*) ' ==================='
*
        WRITE(6,*) ' NORB '
        CALL IWRTMA(NORB,1,8,1,8)
        WRITE(6,*) ' NBAS '
        CALL IWRTMA(NBAS,1,8,1,8)
        WRITE(6,*) 
        WRITE(6,*) ' ECORE = ', ECORE
      END IF
*
      Return
      END
      FUNCTION GIJKLL(IREOTS,IPNTR,ISL,XINT,ISMFTO,IBSO,NACOB,NSMOB,
     &         NOCOBS,I,J,K,L)
*
* Obtain (IJ!KL), Lucas order
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION IREOTS(*),IPNTR(NSMOB,NSMOB,NSMOB)
      DIMENSION ISL(NSMOB,NSMOB,NSMOB)
      DIMENSION IBSO(*),NOCOBS(*) ,ISMFTO(*)
      DIMENSION XINT(*)
*
C?    write(6,*) ' Hi from GIJKLL'
      II = IREOTS(I)
C?    write(6,*) ' II ',II
      ISM = ISMFTO(I)
C?    write(6,*) ' ISM ',ISM
      NI = NOCOBS(ISM)
C?    write(6,*) ' NI ',NI
      IREL = II - IBSO(ISM) + 1
C?    write(6,*) ' IREL ',IREL
      JJ = IREOTS(J)
      JSM = ISMFTO(J)
      JREL = JJ - IBSO(JSM) + 1
      NJ = NOCOBS(JSM)
      IJ = (IREL-1)*NJ + JREL
      JI = (JREL-1)*NI + IREL
      NJI = NI * NJ
      IJSM = (ISM-1)*NSMOB + JSM
*
      KK = IREOTS(K)
      KSM = ISMFTO(K)
      KREL = KK - IBSO(KSM) + 1
      NK = NOCOBS(KSM)
      LL = IREOTS(L)
      LSM = ISMFTO(L)
      LREL = LL - IBSO(LSM) + 1
      NL = NOCOBS(LSM)
      LK = (LREL-1)*NK + KREL
      KL = (KREL-1)*NL + LREL
      NLK = NK * NL
      KLSM = (KSM-1)*NSMOB + LSM
C?    WRITE(6,*) ' IJSM KLSM ', IJSM,KLSM
C?    WRITE(6,*) ' ISM JSM KSM LSM ',ISM,JSM,KSM,LSM
 
      IF(  (IJSM.GE.KLSM.AND.LSM.NE.ISL(ISM,JSM,KSM))
     &.OR. (IJSM.LT.KLSM.AND.JSM.NE.ISL(KSM,LSM,ISM)) )   THEN
        GIJKLL = 0.0D0
        write(6,*) ' Symmetry zero returned '
      ELSE
*
        IJKLO = -2810
        IF(IJSM.GT.KLSM) THEN
C         IJKLO = (IJ-1)*NKL + KL + IPNTR(ISM,JSM,KSM)-1
          IJKLO = (LK-1)*NJI + JI + IPNTR(ISM,JSM,KSM)-1
        ELSE IF(IJSM.LT.KLSM) THEN
C         IJKLO = (KL-1)*NIJ + IJ + IPNTR(KSM,LSM,ISM)-1
          IJKLO = (JI-1)*NLK + LK + IPNTR(KSM,LSM,ISM)-1
        ELSE IF( IJSM.EQ.KLSM) THEN
C         IF(IJ.GE.KL) THEN
          IF(JI.GE.LK) THEN
C           IJKLO = IJ*(IJ-1)/2+KL + IPNTR(ISM,JSM,KSM)-1
            IJKLO = JI*(JI-1)/2+LK + IPNTR(ISM,JSM,KSM)-1
          ELSE
C           IJKLO = KL*(KL-1)/2+IJ + IPNTR(ISM,JSM,KSM)-1
            IJKLO = LK*(LK-1)/2+JI + IPNTR(ISM,JSM,KSM)-1
          END IF
        END IF
        GIJKLL = XINT(IJKLO)
      END IF
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,'(A,5I4,3X,E18.12)')
     &  ' GIJKLL I J K L ,IJKLO,(IJ!KL) ', I,J,K,L,IJKLO,GIJKLL
      END IF
*
      RETURN
      END
      FUNCTION GMIJKL(IORB,JORB,KORB,LORB,INTLST,IJKLOF)
*
* Obtain integral (IORB JORB ! KORB LORB) MOLCAS version
* Integrals assumed in core 
*
* Version for integrals stored in INTLST
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
*. Integral list
      Real * 8 Intlst(*)
      Dimension IJKLOF(NsmOB,NsmOb,NsmOB)
      Logical iSymj,kSyml,ISYMK,JSYML,ijSymkl,IKSYMJL
      Logical ijklPerm
*. 
      NTEST = 00
*
*. The orbital list corresponds to type ordered indeces, reform to
*. symmetry ordering
*
      IABS = IREOTS(IORB)
      ISM = ISMFTO(IORB)
      IOFF = IBSO(ISM)
*
      JABS = IREOTS(JORB)
      JSM = ISMFTO(JORB)
      JOFF = IBSO(JSM)
*
      KABS = IREOTS(KORB)
      KSM = ISMFTO(KORB)
      KOFF = IBSO(KSM)
*
      LABS = IREOTS(LORB)
      LSM = ISMFTO(LORB)
      LOFF = IBSO(LSM)
*
      If( Ntest.ge. 100) THEN
        write(6,*) ' GMIJKL at your service '
        WRITE(6,*) ' IORB IABS ISM IOFF ',IORB,IABS,ISM,IOFF
        WRITE(6,*) ' JORB JABS JSM JOFF ',JORB,JABS,JSM,JOFF
        WRITE(6,*) ' KORB KABS KSM KOFF ',KORB,KABS,KSM,KOFF
        WRITE(6,*) ' LORB LABS LSM LOFF ',LORB,LABS,LSM,LOFF
      END IF
*
      If ( jSm.gt.iSm .or. ( iSm.eq.jSm .and. JABS.gt.IABS)) Then
        iSym=jSm
        jSym=iSm
        I = JABS - JOFF + 1
        J = IABS - IOFF + 1
      Else
        iSym=iSm
        jSym=jSm
        I = IABS - IOFF + 1
        J = JABS - JOFF + 1
      End If
      ijBlk=jSym+iSym*(iSym-1)/2
      If ( lSm.gt.kSm  .or. ( kSm.eq.lSm .and. LABS.gt.KABS)) Then
        kSym=lSm
        lSym=kSm
        K = LABS -LOFF + 1
        L = KABS - KOFF + 1
      Else
        kSym=kSm
        lSym=lSm
        K = KABS - KOFF + 1
        L = LABS -LOFF + 1
      End If
      klBlk=lSym+kSym*(kSym-1)/2
*
      ijklPerm=.false.
      If ( klBlk.gt.ijBlk ) Then
        iTemp=iSym
        iSym=kSym
        kSym=iTemp
        iTemp=jSym
        jSym=lSym
        lSym=iTemp
        iTemp=ijBlk
        ijBlk=klBlk
        klBlk=iTemp
        ijklPerm=.true.
*
        iTemp = i
        i = k
        k = itemp
        iTemp = j
        j = l
        l = iTemp
      End If
      If(Ntest .ge. 100 ) then
        write(6,*) ' i j k l ',i,j,k,l
        write(6,*) ' Isym,Jsym,Ksym,Lsym',Isym,Jsym,Ksym,Lsym
      End if
*
*  Define offset for given symmetry block
      IBLoff = IJKLof(Isym,Jsym,Ksym)
      If(ntest .ge. 100 )
     &WRITE(6,*) ' IBLoff Isym Jsym Ksym ', IBLoff,ISym,Jsym,Ksym
      iSymj=iSym.eq.jSym
      kSyml=kSym.eq.lSym
      iSymk=iSym.eq.kSym
      jSyml=jSym.eq.lSym
      ikSymjl=iSymk.and.jSyml
      ijSymkl=iSymj.and.kSyml
*
      itOrb=NTOOBS(iSym)
      jtOrb=NTOOBS(jSym)
      ktOrb=NTOOBS(kSym)
      ltOrb=NTOOBS(lSym)
C?    print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
      If ( iSymj ) Then
        ijPairs=itOrb*(itOrb+1)/2
        ij=j+i*(i-1)/2
      Else
        ijPairs=itOrb*jtOrb
        ij=j + (i-1)*jtOrb
      End if 
*
      IF(KSYML ) THEN
        klPairs=ktOrb*(ktOrb+1)/2
        kl=l+k*(k-1)/2
      ELSE
        klPairs=ktOrb*ltOrb
        kl=l+(k-1)*ltOrb
      End If
C?    print *,' ijPairs,klPairs',ijPairs,klPairs
*
      If ( ikSymjl ) Then
        If ( ij.gt.kl ) Then
          klOff=kl+(kl-1)*(kl-2)/2-1
          ijkl=ij+(kl-1)*ijPairs-klOff
        Else
          ijOff=ij+(ij-1)*(ij-2)/2-1
          ijkl=kl+(ij-1)*klPairs-ijOff
        End If
      Else
        ijkl=ij+(kl-1)*ijPairs
      End If
      If( ntest .ge. 100 )
     & write(6,*) ' ijkl ', ijkl
*
      GMIJKL = Intlst(iblOff-1+ijkl)
      If( ntest .ge. 100 )
     & write(6,*) ' GMIJKL ', GMIJKL
*
      RETURN
      END 
      SUBROUTINE GRAPW(W,Y,MINEL,MAXEL,NORB,NEL,NTEST)
*
* A graph of strings has been defined from
*
*      MINEL(I) is the smallest allowed number of electrons in
*      orbitals 1 through I
*
*      MAXEL(I) is the largest allowed number of electrons in
*      orbitals 1 through I
*
* Set up vertex weights W
* Set up arc weights    Y
*
* Reverse lexical ordering is used with
* weights of unoccupied orbitals set to 0
*
* Jeppe Olsen
*
       IMPLICIT REAL*8(A-H,O-Z)
       INTEGER W(NORB+1,NEL+1)
       INTEGER Y(NORB,NEL)
       INTEGER MAXEL(NORB),MINEL(NORB)
*
C      NTEST = 0
       CALL ISETVC(W,0,(NEL+1)*(NORB+1) )
       CALL ISETVC(Y,0,NEL*NORB)
*
*================
*  Vertex weights
*================
*
*. (Weight for vertex(IEL,IORB) is stored in W(IORB+1,IEL+1) )
      W(1,1) = 1
      DO 300 IEL = 0, NEL
        DO 200 IORB = 1, NORB
          IF(MINEL(IORB).LE.IEL .AND. IEL .LE. MAXEL(IORB) ) THEN
            IF( IEL .GT. 0 ) THEN
              W(IORB+1,IEL+1) = W(IORB-1+1,IEL+1)
     &                        + W(IORB-1+1,IEL-1+1)
            ELSE
              W(IORB+1,1) = W(IORB-1+1,1)
            END IF
          END IF
  200   CONTINUE
  300 CONTINUE
*
*=============
* Arc weights
*=============
*
*. Weight for arc connecting vertices (IORB-1,IEL-1) and(IORB,IEL)
*. is stored in Y(IORB,IEL)
*. Y(IORB,IEL) = W(IORB-1,IEL)
      DO 1300 IEL = 1, NEL
        DO 1200 IORB = 1, NORB
          IF(MINEL(IORB).LE.IEL .AND. IEL .LE. MAXEL(IORB) ) THEN
            Y(IORB,IEL) = W(IORB-1+1,IEL+1)
          END IF
 1200   CONTINUE
 1300 CONTINUE
*
      IF( NTEST .GE.10 ) THEN
C       WRITE(6,'(A)') ' Matrix of vertex weights '
C       WRITE(6,'(A)') ' ========================'
C       CALL IWRTMA(W,NORB+1,NEL+1,NORB+1,NEL+1)
        WRITE(6,'(A)') '  Matrix for arc weights  '
        WRITE(6,'(A)') '  ======================'
        CALL IWRTMA(Y,NORB,NEL,NORB,NEL)
      END IF
*
      RETURN
      END
      SUBROUTINE GSTTBL(C,CTT,IATP,IASM,IBTP,IBSM,IOCOC,
     &                  NOCTPA,NOCTPB,NSASO,NSBSO,PSSIGN,ICOOSC,IDC,
     &                  PLSIGN,LUC,SCR,NSMST,ISCALE,SCLFAC)
*
* obtain  determinant block (iatp iasm, ibtp ibsm )
* from vector packed in combination format according to IDC
*
*. If ISCALE = 1, the routine scales and returns the block
*  in determinant normalization, and SCLFAC = 1.0D0
*
* If ISCALE = 0, the routine does not perform any overall
* scaling, and a scale factor is returned in SCLFAC
*
* IF ISCALE = 0, zero blocks are not set explicitly to zero,
* instead  zero is returned in SCLFAC
*
* IF LUC .lt. 0, then packed vector is assumed to already be in 
* C
*
* ISCALE, SCLFAC added May 97
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION C(*),CTT(*),NSASO(NSMST, *),NSBSO(NSMST, *)
      DIMENSION IOCOC(NOCTPA,NOCTPB),ICOOSC(NOCTPA,NOCTPB,*)
      DIMENSION SCR(*)
      REAL*8 INPROD
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        write(6,*) ' GSTTBL  ,IATP,IASM,IBTP,IBSM,ISCALE'
        write(6,*)            IATP,IASM,IBTP,IBSM,ISCALE     
        WRITE(6,*) ' PSSIGN, PLSIGN = ', PSSIGN, PLSIGN
      END IF
*
      NAST = NSASO(IASM,IATP)
      NBST = NSBSO(IBSM,IBTP)
      IF(IDC.EQ.1.OR. .NOT.(IATP.EQ.IBTP.AND.IASM.EQ.IBSM)) THEN 
       NCMB_IN = NAST*NBST
      ELSE
       NCMB_IN = NAST*(NAST+1)/2
      END IF
*
      IF(LUC.GT.0) THEN
        CALL IFRMDS(LBL,1,-1,LUC)
C?      write(6,*) ' LBL = ', LBL
        IF(ISCALE.EQ.1) THEN
           CALL FRMDSC(SCR,LBL,-1,LUC,IMZERO,IAMPACK)
        ELSE
          NO_ZEROING = 1
          CALL FRMDSC2(SCR,LBL,-1,LUC,IMZERO,IAMPACK,NO_ZEROING)
        END IF
      ELSE
*. It is assumed pt that vector is nonvanising
        XNORM = INPROD(C,C,NCMB_IN)
C?      WRITE(6,*) ' XNORM in GSTT ', XNORM
        IF(XNORM.EQ.0.0D0) THEN
          IMZERO = 1
        ELSE 
          IMZERO = 0
        END IF
        LBL = NCMB_IN
      END IF
*
       IF(IMZERO.EQ.1.AND.ISCALE.EQ.0) THEN
         SCLFAC = 0.0D0
       ELSE
          NAST = NSASO(IASM,IATP)
          NBST = NSBSO(IBSM,IBTP)
          IF(LBL.NE.0) THEN
            PLSIGN = 1.0D0
            ISGVST = 1 
            IF(LUC.GT.0) THEN
              CALL SDCMRF(CTT,SCR,2,IATP,IBTP,IASM,IBSM,NAST,NBST,
     &             IDC,PSSIGN,PLSIGN,ISGVST,LDET,LCOMB,ISCALE,SCLFAC)
            ELSE
              CALL SDCMRF(CTT,C,2,IATP,IBTP,IASM,IBSM,NAST,NBST,
     &             IDC,PSSIGN,PLSIGN,ISGVST,LDET,LCOMB,ISCALE,SCLFAC)
            END IF
          ELSE
           SCLFAC = 0.0D0
          END IF
        END IF
*
        IF(NTEST.GE.100) THEN
         WRITE(6,*) ' ISCALE, IMZERO, SCLFAC on return in GSTTBL',
     &   ISCALE, IMZERO, SCLFAC  
        END IF
*
      RETURN
      END
      SUBROUTINE GT1DIA(H1DIA)
*
* Obtain diagonal of one electron matrix over all
* orbitals
*
*. Dec 97: obtained from KINT1O
*. June 2010: Changed back to KINT1!
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
#include "mafdecls.fh"
#include "global.fh"
      INCLUDE 'wrkspc.inc'
 
*.GLobal pointers
C     COMMON/GLBBAS/KINT1,KINT2,KPINT1,KPINT2,KLSM1,KLSM2,KRHO1
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
*
CINA  CALL GT1DIS(H1DIA,IREOTS(1+NINOB),WORK(KPINT1),WORK(KINT1),
CINA &            ISMFTO,IBSO,NACOB)
COLD  CALL GT1DIS(H1DIA,IREOTS(1),WORK(KPINT1),WORK(KINT1O),
COLD &            ISMFTO,IBSO,NTOOB)
c..dongxia
c..  work(kpint1) and work(kint1) are static arrays about integrals
c..  may need to convert to GA. Leave here for the time being.
      CALL GT1DIS(H1DIA,IREOTS(1),WORK(KPINT1),WORK(KINT1),
     &            ISMFTO,IBSO,NTOOB)
*
      RETURN
      END
      SUBROUTINE GT1DIS(H1DIA,IREOTS,IPNT,H,ISMFTO,IBSO,NTOOB)
*
* diagonal of one electron integrals over all orbitals in type order
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      INTEGER IREOTS(*),IPNT(*),ISMFTO(*),IBSO(*)
      DIMENSION H(*)
*.Output
      DIMENSION H1DIA(*)
*
      DO 100 IIOB = 1, NTOOB
        IOB = IREOTS(IIOB)
        ISM = ISMFTO(IIOB)
        IOBREL = IOB-IBSO(ISM)+1
C?      WRITE(6,*) ' IIOB IOB ISM IOBREL '
C?      WRITE(6,*)   IIOB,IOB,ISM,IOBREL
        H1DIA(IIOB) = H(IPNT(ISM)-1+IOBREL*(IOBREL+1)/2)
  100 CONTINUE
*
      NTEST = 00
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Diagonal one electron integrals '
        CALL WRTMAT(H1DIA,1,NTOOB,1,NTOOB)
      END IF
*
      RETURN
      END
      FUNCTION GTH1ES(IREOTS,IPNT,H,IBSO,MXPNGAS,
     &           IBTSOB_GN,NTOOBS,IORB,ITP,ISM,JORB,JTP,JSM,IJSM,
     &           NINOB)
*
* one electron integral between orbitals (iorb,itp,ism,jorb,jsm,jtp)
* correct combination of row and column symmetry is assumed
* IJSM = 1 => Lower triangular packed
*      else=> No triangular packing
*
* Last Revision January 98 (IJSM added )
* July 2010: IBTSOB_GN replacing IBTSOB
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      INTEGER IREOTS(*),IPNT(*),IBTSOB_GN(0:MXPNGAS,*),IBSO(*)
      INTEGER NTOOBS(*)
      DIMENSION H(*)
*
      NTEST = 0
*
      IABS = IORB+IBTSOB_GN(ITP,ISM)-1
      IREO = IREOTS(IABS)
      JABS = JORB+IBTSOB_GN(JTP,JSM)-1
      JREO = IREOTS(JABS)
*
      IF(NTEST.GE.100) THEN
*
        write(6,'(A,6I4)') ' GTH1ES: IORB, ITP, ISM, JORB, JTP, JSM ',
     &                               IORB, ITP, ISM, JORB, JTP, JSM
        write(6,'(A,4I5)') ' GTH1ES: IABS, JABS, IREO JREO ',
     &                       IABS, JABS IREO,JREO
        write(6,*) ' GTH1ES: IBSO ', IBSO(ISM)
*
      END IF
*
      IJ = -2303
      IF(IJSM.EQ.1) THEN
        IF(ISM.GT.JSM) THEN
          NI = NTOOBS(ISM)
          IJ = IPNT(ISM)-1+(JREO-IBSO(JSM))*NI+IREO-IBSO(ISM)+1
        ELSE IF(ISM.EQ.JSM) THEN
          IJMAX = MAX(IREO-IBSO(ISM)+1,JREO-IBSO(JSM)+1)
          IJMIN = MIN(IREO-IBSO(ISM)+1,JREO-IBSO(JSM)+1)
          IJ = IPNT(ISM)-1+IJMAX*(IJMAX-1)/2+IJMIN
        ELSE IF (ISM.LT.JSM) THEN
          NJ = NTOOBS(JSM)
          IJ = IPNT(JSM)-1+(IREO-IBSO(ISM))*NJ+JREO-IBSO(JSM)+1
        END IF
      ELSE 
         NI = NTOOBS(ISM)
         IJ = IPNT(ISM)-1+(JREO-IBSO(JSM))*NI+IREO-IBSO(ISM)+1
      END IF
*
      GTH1ES = H(IJ)
*
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' One electron integral from GTH1ES '
        WRITE(6,'(A,2I4)') ' IJSM,IPNT(ISM) ', IJSM,IPNT(ISM)
        WRITE(6,'(A,5I4)') ' IABS, IREO, IORB, ITP, ISM ',
     &                       IABS, IREO, IORB, ITP, ISM
        WRITE(6,'(A,5I4)') ' JABS, JREO, JORB, JTP, JSM',
     &                       JABS, JREO, JORB, JTP, JSM
        WRITE(6,'(A,I5,E22.15)') ' IJ and H(IJ) ', IJ,H(IJ)
      END IF
*
      RETURN
      END
      FUNCTION GTIJKL_OLD(I,J,K,L)
*
* Obtain  integral (I J ! K L )
* where I,J,K and l refers to active orbitals in
* Type ordering
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'oper.inc'
*
      XIJKL = 0.0D0
      IF(INTIMP .EQ. 2 ) THEN
*. LUCAS ordering
        I12S = 0
        I34S = 0
        I1234S = 1
        XIJKL = GIJKLL(IREOTS(1),WORK(KPINT2),WORK(KLSM2),
     &           WORK(KINT2),
     &           ISMFTO,IBSO,NACOB,NSMOB,NOCOBS,I,J,K,L)
      ELSE IF (INTIMP.EQ.1.OR.INTIMP.EQ.5) THEN
*. MOLCAS OR SIRIUS IMPORT ( I hope integrals are in core !! )
          IF(I_USE_SIMTRH.EQ.0) THEN
            XIJKL = GMIJKL(I,J,K,L,WORK(KINT2),WORK(KPINT2))
          ELSE IF(I_USE_SIMTRH.EQ.1) THEN
            IADR =  I2EAD_NOCCSYM(IREOTS(I),IREOTS(J),
     &              IREOTS(K),IREOTS(L),0)
            XIJKL = WORK(KINT2_SIMTRH-1+IADR)
          END IF
      END IF
      GTIJKL_OLD = XIJKL
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Integral for I,J,K,L = ', I,J,K,L, ' is ', XIJKL
      END IF
*
      RETURN
      END
      SUBROUTINE GTJK(RJ,RK,NTOOB,SCR,IREOTS)
*
* Interface routine for obtaining Coulomb (RJ) and
* Exchange integrals (RK) from the current active list of 
* integrals
*
* Ordering of integrals is the internal order (type)
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'crun.inc'
*.Input
      DIMENSION IREOTS(*)
*.Output
      DIMENSION RJ(NTOOB,NTOOB),RK(NTOOB,NTOOB)
*.Scratch: Is not needed !!!
      DIMENSION SCR(2)
*. Call the slave 
cGLM      CALL GTJKS(RJ,RK,NTOOB)
      CALL GTJKL(RJ,RK,NTOOB)
      NTEST = 00
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' RJ and RK from GTJK '
        CALL WRTMAT(RJ,NTOOB,NTOOB,NTOOB,NTOOB)
        CALL WRTMAT(RK,NTOOB,NTOOB,NTOOB,NTOOB)
      END IF
*
      RETURN
      END
      SUBROUTINE GTJKL(RJ,RK,NTOOB)
*
* Obtain Coulomb  integrals (II!JJ)
*        exchange integrals (IJ!JI)
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION RJ(NTOOB,NTOOB),RK(NTOOB,NTOOB)
*
      DO 100 IORB = 1, NTOOB
        DO 50 JORB = 1, NTOOB
          RJ(IORB,JORB) = GTIJKL(IORB,IORB,JORB,JORB)
          RK(IORB,JORB) = GTIJKL(IORB,JORB,JORB,IORB)
   50   CONTINUE
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' RJ and RK from GTJK '
        CALL WRTMAT(RJ,NTOOB,NTOOB,NTOOB,NTOOB)
        CALL WRTMAT(RK,NTOOB,NTOOB,NTOOB,NTOOB)
      END IF
*
      RETURN
      END
      SUBROUTINE GTJKS(J,K,NORB)
*
* Obtain Coulomb and Exchange integrals
* from complete integral list stored in core
*
      IMPLICIT REAL*8           (A-H,O-Z)
      REAL*8           J(NORB,NORB),K(NORB,NORB)
*	
      DO 200 IORB = 1, NORB
        DO 100 JORB = 1, NORB
         J(IORB,JORB) = GTIJKL_GN(IORB,IORB,JORB,JORB)
         K(IORB,JORB) = GTIJKL_GN(IORB,JORB,JORB,IORB)
cGLM         J(IORB,JORB) = GTIJKL(IORB,IORB,JORB,JORB)
cGLM         K(IORB,JORB) = GTIJKL(IORB,JORB,JORB,IORB)
  100   CONTINUE
  200 CONTINUE
*
      NTEST = 10
      IF(NTEST.NE.0) THEN
        write(6,*) 'IORB IORB JORB JORB, J(IORB,JORB)'
        DO IORB = 1, NORB
          DO JORB = 1, NORB
            write(6,*) IORB, IORB, JORB, JORB, J(IORB,JORB)
          END DO  
        END DO
        write(6,*) 'IORB IORB JORB JORB, K(IORB,JORB)'
        DO IORB = 1, NORB
          DO JORB = 1, NORB
            write(6,*) IORB, JORB, JORB, IORB, K(IORB,JORB)
          END DO  
        END DO
*        WRITE(6,*) ' RJ from GTJKL '
*        CALL WRTMAT(RJ,NTOOB,NTOOB,NTOOB,NTOOB)
*        WRITE(6,*) ' RK from GTJKL '
*        CALL WRTMAT(RK,NTOOB,NTOOB,NTOOB,NTOOB)
      END IF
      RETURN
      END

      Subroutine GTJKM(RJ,RK)
*
*     Gather all integrals RJ(I,J) = (II!JJ)
*     Gather all integrals RK(I,J) = (IJ!IJ)
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
*.LUNDIO
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
      INTEGER*8 iDAdr
*.Output
      DIMENSION RJ(NTOOB,NTOOB),RK(NTOOB,NTOOB)
*.Local
      Parameter ( lBuf    = 9600 )
      Dimension Scr(lBuf)
*

      write(6,*) ' ****************************** '
      write(6,*) ' >>>>>>> Input to GTJKM <<<<<<< '
      write(6,*) ' ****************************** '
      write(6,*) ' nSmOb        :', nSmOb
      write(6,*) ' NTOOBS(iSym) :', (NTOOBS(i),i =1, nSmOb)
      write(6,*) ' LuTr2        :', LuTr2
      write(6,*) ' lTr2Rec      :', lTr2Rec
      write(6,*) ' iDAdr        :', iDAdr
      write(6,*) ' ****************************** '
      Do iSym=1,nSmOb
        itOrb=NTOOBS(iSym)
        itOff=ITOOBS(iSym)
        iiBlk=iSym*(iSym+1)/2
        iiPairs=itOrb*(itOrb+1)/2
        Do jSym=1,iSym
          jtOrb=NTOOBS(jSym)
          jtOff=ITOOBS(jSym)
          jjBlk=jSym*(jSym+1)/2
          jjPairs=jtOrb*(jtOrb+1)/2
          ijPairs=itOrb*jtOrb
          If ( iSym.eq.jSym ) ijPairs=jtOrb+itOrb*(itOrb-1)/2
          ijBlk=jSym+iSym*(iSym-1)/2
*
*     collect all RJ(iOrb,jOrb)=(II,JJ)
*
          iRecOld=-1
          iSyBlk=jjBlk+iiBlk*(iiBlk-1)/2
          iBatch=nBatch(iSyBlk)
          iDisk=iDAdr(iBatch)
          nInts=iiPairs*jjPairs
          Do i=1,itOrb
            ii=i*(i+1)/2
*JOS
            MaxJ = jtOrb
            If(Isym.eq.Jsym) MaxJ = i
            Do j=1,MaxJ
*JOE
              jj=j*(j+1)/2
              iijj=ii+(jj-1)*iiPairs
*JOS
              If ( Isym.eq.Jsym ) Then
                jjOff=jj+(jj-1)*(jj-2)/2-1
                iijj =  iijj - jjOff
              End If
*JOE
              iRec=(iijj-1)/lTr2Rec
              If ( iRec.eq.iRecOld ) then
                iijj=iijj-iRec*lTr2Rec
              Else
                iDisk=iDAdr(iBatch)
                Do iSkip=1,iRec
cGLM                  Call DaFile(LuTr2,0,Scr,2*lTr2Rec,iDisk)
                  Call dDaFile(LuTr2,0,Scr,lTr2Rec,iDisk)
                End Do
cGLM                Call DaFile(LuTr2,2,Scr,2*lTr2Rec,iDisk)
                Call dDaFile(LuTr2,2,Scr,lTr2Rec,iDisk)
                iijj=iijj-iRec*lTr2Rec
                iRecOld=iRec
              End If
              RJ(i+itOff-1,j+jtOff-1)=Scr(iijj)
              RJ(j+jtOff-1,i+itOff-1)=Scr(iijj)
            End Do
          End Do
*
*     collect all RK(iOrb,jOrb)=(IJ,IJ)
*
          iRecOld=-1
          iSyBlk=ijBlk*(ijBlk+1)/2
          iBatch=nBatch(iSyBlk)
          iDisk=iDAdr(iBatch)
          nInts=ijPairs*(ijPairs+1)/2
          ij=0
          Do i=1,itOrb
            jMax=jtOrb
            If ( iSym.eq.jSym ) jMax=i
            Do j=1,jMax
              ij=ij+1
              ijOff=ij+(ij-1)*(ij-2)/2-1
              ijij=ij+(ij-1)*ijPairs-ijOff
              iRec=(ijij-1)/lTr2Rec
              If ( iRec.eq.iRecOld ) then
                ijij=ijij-iRec*lTr2Rec
              Else
                iDisk=iDAdr(iBatch)
                Do iSkip=1,iRec
cGLM                  Call DaFile(LuTr2,0,Scr,2*lTr2Rec,iDisk)
                  Call dDaFile(LuTr2,0,Scr,lTr2Rec,iDisk)
                End Do
*                Call DaFile(LuTr2,2,Scr,2*lTr2Rec,iDisk)
                Call dDaFile(LuTr2,2,Scr,lTr2Rec,iDisk)
                ijij=ijij-iRec*lTr2Rec
                iRecOld=iRec
              End If
              RK(i+itOff-1,j+jtOff-1)=Scr(ijij)
              RK(j+jtOff-1,i+itOff-1)=Scr(ijij)
            End Do
          End Do
*
        End Do
      End Do
*
      RETURN
      END

* Working on EXPHAM
* some known problems:
*     1: if CSF are used diagonal is not delivered to H0mat
*      SUBROUTINE GTJKS(J,K,NORB)
*
* Obtain Coulomb and Exchange integrals
* from complete integral list stored in core
*
*      IMPLICIT REAL*8           (A-H,O-Z)
*      REAL*8           J(NORB,NORB),K(NORB,NORB)
*      DO 200 IORB = 1, NORB
*	DO 100 JORB = 1, NORB
*	  J(IORB,JORB) = GTIJKL_GN(IORB,IORB,JORB,JORB)
*	  K(IORB,JORB) = GTIJKL_GN(IORB,JORB,JORB,IORB)
*  100   CONTINUE
*  200 CONTINUE
**
*      RETURN
*      END 
      FUNCTION I2EAD(IORB,JORB,KORB,LORB)
*
* Find adress of integral in LUCIA order 
*
*
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
*
*
      I2EAD = I2EADS(IORB,JORB,KORB,LORB,WORK(KPINT2))
*
      RETURN
      END
      FUNCTION I2EADS(IORB,JORB,KORB,LORB,IJKLOF)
*
* Obtain address of integral (IORB JORB ! KORB LORB) in MOLCAS order 
* IORB JORB KORB LORB corresponds to SYMMETRY ordered indeces !!
* Integrals assumed in core 
*
* The new ordering of integrals (flagged by ITRA_ROUTE in crun) is allowed per June 2011
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'crun.inc'
*
      Dimension IJKLOF(NsmOB,NsmOb,NsmOB)
      Logical iSymj,kSyml,ISYMK,JSYML,ijSymkl,IKSYMJL
      Logical ijklPerm
*. 
      NTEST = 00
*
      IABS = IORB
      ISM = ISMFTO(IREOST(IORB))
      IOFF = IBSO(ISM)
*
      JABS = JORB
      JSM = ISMFTO(IREOST(JORB))
      JOFF = IBSO(JSM)
*
      KABS = KORB
      KSM = ISMFTO(IREOST(KORB))
      KOFF = IBSO(KSM)
*
      LABS = LORB
      LSM = ISMFTO(IREOST(LORB))
      LOFF = IBSO(LSM)
*
      If( Ntest.ge. 100) THEN
        write(6,*) ' I2EADS at your service '
        WRITE(6,*) ' IORB IABS ISM IOFF ',IORB,IABS,ISM,IOFF
        WRITE(6,*) ' JORB JABS JSM JOFF ',JORB,JABS,JSM,JOFF
        WRITE(6,*) ' KORB KABS KSM KOFF ',KORB,KABS,KSM,KOFF
        WRITE(6,*) ' LORB LABS LSM LOFF ',LORB,LABS,LSM,LOFF
      END IF
*
      If ( jSm.gt.iSm .or. ( iSm.eq.jSm .and. JABS.gt.IABS)) Then
        iSym=jSm
        jSym=iSm
        I = JABS - JOFF + 1
        J = IABS - IOFF + 1
      Else
        iSym=iSm
        jSym=jSm
        I = IABS - IOFF + 1
        J = JABS - JOFF + 1
      End If
      ijBlk=jSym+iSym*(iSym-1)/2
      If ( lSm.gt.kSm  .or. ( kSm.eq.lSm .and. LABS.gt.KABS)) Then
        kSym=lSm
        lSym=kSm
        K = LABS -LOFF + 1
        L = KABS - KOFF + 1
      Else
        kSym=kSm
        lSym=lSm
        K = KABS - KOFF + 1
        L = LABS -LOFF + 1
      End If
      klBlk=lSym+kSym*(kSym-1)/2
*
      ijklPerm=.false.
      If ( klBlk.gt.ijBlk ) Then
        iTemp=iSym
        iSym=kSym
        kSym=iTemp
        iTemp=jSym
        jSym=lSym
        lSym=iTemp
        iTemp=ijBlk
        ijBlk=klBlk
        klBlk=iTemp
        ijklPerm=.true.
*
        iTemp = i
        i = k
        k = itemp
        iTemp = j
        j = l
        l = iTemp
      End If
      If(Ntest .ge. 100 ) then
        write(6,*) ' i j k l ',i,j,k,l
        write(6,*) ' Isym,Jsym,Ksym,Lsym',Isym,Jsym,Ksym,Lsym
      End if
*
*  Define offset for given symmetry block
      IBLoff = IJKLof(Isym,Jsym,Ksym)
      If(ntest .ge. 100 )
     &WRITE(6,*) ' IBLoff Isym Jsym Ksym ', IBLoff,ISym,Jsym,Ksym
      iSymj=iSym.eq.jSym
      kSyml=kSym.eq.lSym
      iSymk=iSym.eq.kSym
      jSyml=jSym.eq.lSym
      ikSymjl=iSymk.and.jSyml
      ijSymkl=iSymj.and.kSyml
*
      itOrb=NTOOBS(iSym)
      jtOrb=NTOOBS(jSym)
      ktOrb=NTOOBS(kSym)
      ltOrb=NTOOBS(lSym)
C?    print *,' itOrb,jtOrb,ktOrb,ltOrb',itOrb,jtOrb,ktOrb,ltOrb
      If ( iSymj ) Then
        ijPairs=itOrb*(itOrb+1)/2
        ij=j+i*(i-1)/2
      Else
        ijPairs=itOrb*jtOrb
        IF(ITRA_ROUTE.EQ.1) THEN
          ij=j + (i-1)*jtOrb
        ELSE
          ij=I + (J-1)*ItOrb
        END IF
      End if 
*
      IF(KSYML ) THEN
        klPairs=ktOrb*(ktOrb+1)/2
        kl=l+k*(k-1)/2
      ELSE
        klPairs=ktOrb*ltOrb
        IF(ITRA_ROUTE.EQ.1) THEN
          kl=l+(k-1)*ltOrb
        ELSE
          kl=K+(L-1)*KtOrb
        END IF
      End If
C?    print *,' ijPairs,klPairs',ijPairs,klPairs
*
      If ( ikSymjl ) Then
        If ( ij.gt.kl ) Then
          klOff=kl+(kl-1)*(kl-2)/2-1
          ijkl=ij+(kl-1)*ijPairs-klOff
        Else
          ijOff=ij+(ij-1)*(ij-2)/2-1
          ijkl=kl+(ij-1)*klPairs-ijOff
        End If
      Else
        ijkl=ij+(kl-1)*ijPairs
      End If
      If( ntest .ge. 100 )
     & write(6,*) ' ijkl ', ijkl
*
      I2EADS = iblOff-1+ijkl
      If( ntest .ge. 100 ) then
        write(6,*) 'i j k l ', i,j,k,l
        write(6,*) ' ibloff ijkl ',ibloff,ijkl
        write(6,*) ' I2EADS  = ', I2EADS
      END IF
*
      RETURN
      END 
      FUNCTION IABNUM(IASTR,IBSTR,IAGRP,IBGRP,IGENSG,
     &                ISGNA,ISGNB,ISGNAB,IOOS,NORB,IPSFAC,PSSIGN,
     &                IPRNT)
*
* Encapsulation routine for IABNUS
*
c      IMPLICIT REAL*8           (A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      DIMENSION IASTR(*),IBSTR(*)
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'strbas.inc'
*
      IABNUM = IABNUS(IASTR,NELEC(IAGRP),WORK(KSTREO(IAGRP)),
     &         WORK(KSTCL(IAGRP)),WORK(KSTSM(IAGRP)),NOCTYP(IAGRP),
     &         WORK(KZ(IAGRP)),WORK(KISTSO(IAGRP)),int_mb(KNSTSO(IAGRP)),
     &                IBSTR,NELEC(IBGRP),WORK(KSTREO(IBGRP)),
     &         WORK(KSTCL(IBGRP)),WORK(KSTSM(IBGRP)),NOCTYP(IBGRP),
     &         WORK(KZ(IBGRP)),WORK(KISTSO(IBGRP)),int_mb(KNSTSO(IBGRP)),
     &         IOOS,NORB,IGENSG,ISGNA,ISGNAB,ISGNAB,PSSIGN,IPSFAC,
     &         IPRNT)
      RETURN
      END
      FUNCTION IABNUS(IASTR,NAEL,IAORD,ITPFSA,ISMFSA,NOCTPA,ZA,
     &                ISSOA,NSSOA,
     &                IBSTR,NBEL,IBORD,ITPFSB,ISMFSB,NOCTPB,ZB,
     &                ISSOB,NSSOB,
     &                IOOS,NORB,IGENSG,ISGNA,ISGNB,ISGNAB,
     &                PSSIGN,IPSFAC,IPRNT)
*
* A determinant is given by strings IASTR,IBSTR .
* Find number of this determinant
*
* If PSSIGN .ne. 0, the determinant with higher alpha number is picked
* and phase factor IPSFAC calculated. This corresponds to
* configuration order
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION IASTR(NAEL),IBSTR(NBEL)
      DIMENSION IAORD(*),IBORD(*)
      INTEGER ZA(*),ZB(*)
      INTEGER NSSOA(NOCTPA,*),NSSOB(NOCTPB,*)
      INTEGER ISSOA(NOCTPA,*),ISSOB(NOCTPB,*)
      INTEGER IOOS(NOCTPA,NOCTPB,*)
      INTEGER ISGNA(*),ISGNB(*)
      INTEGER ITPFSA(*),ITPFSB(*)
      INTEGER ISMFSA(*),ISMFSB(*)
*
* Jeppe Olsen
*
      NTEST =  00
      NTEST = MAX(NTEST,IPRNT)
      IF( NTEST .GT. 300) THEN
       WRITE(6,*) ' >>> IABNUS SPEAKING <<< '
       WRITE(6,*) ' NOCTPA,NOCTPB ', NOCTPA,NOCTPB
       WRITE(6,*) ' ALPHA AND BETA STRING '
       CALL IWRTMA(IASTR,1,NAEL,1,NAEL)
       CALL IWRTMA(IBSTR,1,NBEL,1,NBEL)
      END IF
*.Number of alpha- and beta-string
C             ISTRNM(IOCC,NORB,NEL,Z,NEWORD,IREORD)
      IANUM = ISTRNM(IASTR,NORB,NAEL,ZA,IAORD,1)
      IBNUM = ISTRNM(IBSTR,NORB,NBEL,ZB,IBORD,1)
      IF( NTEST .GE. 10 ) WRITE(6,*) ' IANUM AND IBNUM ',IANUM,IBNUM
*
      IF(IGENSG.NE.0) THEN
        ISGNAB = ISGNA(IANUM)*ISGNB(IBNUM)
      ELSE
        ISGNAB = 1
      END IF
*. Symmetries and types
      IASYM = ISMFSA(IANUM)
      IBSYM = ISMFSB(IBNUM)
C?    IF( NTEST .GE.10) WRITE(6,*) ' IASYM IBSYM ',IASYM,IBSYM
      IATP = ITPFSA(IANUM)
      IBTP = ITPFSB(IBNUM)
C?    IF(NTEST.GE.10) WRITE(6,*) ' IATP,IBTP ', IATP,IBTP
      IAREL = IANUM - ISSOA(IATP,IASYM)+1
      IBREL = IBNUM - ISSOB(IBTP,IBSYM)+1
C?    IF(NTEST .GE.10) WRITE(6,*) ' IAREL IBREL ', IAREL,IBREL
*
      IF(PSSIGN.EQ.0) THEN
*.      Normal determinant ordering
        IABNUS = IOOS(IATP,IBTP,IASYM)
     &         + (IBREL-1)*NSSOA(IATP,IASYM) + IAREL - 1
        IPSFAC = 1
      ELSE IF (PSSIGN .NE. 0 ) THEN
*.      Ensure mapping to proper determinant in combination
        IF(IANUM.GE.IBNUM) THEN
*.        No need for switching around so
          IF(IASYM.EQ.IBSYM .AND. IATP. EQ. IBTP ) THEN
*.          Lower triangular packed, column wise !
            IABNUS = IOOS(IATP,IBTP,IASYM)  -1
     &             + (IBREL-1)*NSSOA(IATP,IASYM) + IAREL
     &             -  IBREL*(IBREL-1)/2
          ELSE
            IABNUS = IOOS(IATP,IBTP,IASYM)
     &             + (IBREL-1)*NSSOA(IATP,IASYM) + IAREL - 1
          END IF
          IPSFAC = 1
        ELSE IF (IBNUM .GT. IANUM ) THEN
*. Switch alpha and beta string around
          IF(IASYM.EQ.IBSYM .AND. IATP. EQ. IBTP ) THEN
*. Lower triangular packed, column wise !
            IABNUS = IOOS(IBTP,IATP,IBSYM)  -1
     &             + (IAREL-1)*NSSOB(IBTP,IBSYM) + IBREL
     &             -  IAREL*(IAREL-1)/2
          ELSE
            IABNUS = IOOS(IBTP,IATP,IBSYM)
     &             + (IAREL-1)*NSSOB(IBTP,IBSYM) + IBREL
     &             -  1
          END IF
          IPSFAC = PSSIGN
        END IF

      END IF
*
COLD
COLD    IABNUS = IOOS(IATP,IBTP,IASYM) + (IBREL-1)*NSSOA(IATP,IASYM)
COLD &           + IAREL - 1
C?    IF(NTEST .GT. 10 ) then
C?      WRITE(6,*) ' IOOS NSSOA ',IOOS(IATP,IBTP,IASYM),
C?   &              NSSOA(IATP,IASYM)
C?    END IF
*
      IF ( NTEST .GE.200) THEN
         WRITE(6,*) ' ALPHA AND BETA STRING '
         CALL IWRTMA(IASTR,1,NAEL,1,NAEL)
         CALL IWRTMA(IBSTR,1,NBEL,1,NBEL)
         WRITE(6,*) ' Corresponding determinant number ', IABNUS
      END IF
*
      RETURN
      END
      SUBROUTINE IAIBCM_GAS(LCMBSPC,ICMBSPC,
     &           MNMXOC,NOCTPA,NOCTPB,IOCA,IOCB,NELFTP,
     &           MXPNGAS,NGAS,IOCOC,IPRNT,I_RE_MS2_SPACE,
     &           I_RE_MS2_VALUE,I_CHECK_ENSGSOCC)
*
* Allowed combinations of alpha and beta types, GAS version
*
*
* =====
*.Input
* =====
*
* LCMBSPC: Number of GAS spaces included in this expnasion
* ICMBSPC: Gas spaces included in this expansion 
*
* MXMNOC(IGAS,1,IGASSPC): Min accumulated occ for AS 1-IGAS for space IGASSPC
* MXMNOC(IGAS,2,IGASSPC): Max accumulated occ for AS 1-IGAS for space IGASSPC
*
* NOCTPA: Number of alpha types 
* NOCTPB: Number of beta types
*
* IOCA(IGAS,ISTR) occupation of AS IGAS for alpha string type ISTR
* IOCB(IGAS,ISTR) occupation of AS IGAS for beta  string type ISTR
*
* MXPNGAS: Largest allowed number of gas spaces 
* NGAS   : Actual number of gas spaces
*              IENSGS,LENSGS,NELVAL_IN_ENSGS,IEL_IN_ENSGS)
* LENSGS: Number of Gaspaces in ensemble GASpace
* IENSGS: The gas spaces in the ensemble space
*
* If ICHECK_ENSGS_OCC .eq.1. then the occupation in the 
* ensemble of gasorbitals is checked

      
*
* ======
*.Output
* ======
*
* IOCOC(IATP,IBTP)  = 1 =>      allowed combination
* IOCOC(IATP,IBTP)  = 0 => not allowed combination
*
*.Input
      INTEGER ICMBSPC(LCMBSPC)
      INTEGER MNMXOC(MXPNGAS,2,*)
C     INTEGER MNOCC(NGAS),MXOCC(NGAS)            
      INTEGER IOCA(MXPNGAS,NOCTPA),IOCB(MXPNGAS,NOCTPB)
      INTEGER NELFTP(*)
*.Output
      INTEGER IOCOC(NOCTPA,NOCTPB)
*. Local scratch: occ per gaspace
      INTEGER IGSOCC(100)
*
      NTEST = 00
      NTEST = MAX(NTEST,IPRNT)
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' IAIBCM_GAS entered '
        WRITE(6,*) ' ==================='
        WRITE(6,*) 
        WRITE(6,*) ' Number of GAS spaces included ', LCMBSPC
        WRITE(6,*) ' GAS spaces included ',(ICMBSPC(II),II=1,LCMBSPC)
        WRITE(6,*)
        WRITE(6,*) ' I_CHECK_ENSGSOCC = ', I_CHECK_ENSGSOCC
        IF(NTEST.GE.200) THEN
          WRITE(6,*) ' IOCA and IOCB '
          CALL IWRTMA(IOCA,NGAS,NOCTPA,MXPNGAS,NGAS)
          CALL IWRTMA(IOCB,NGAS,NOCTPB,MXPNGAS,NGAS)
        END IF
      END IF
*
      CALL ISETVC(IOCOC,0,NOCTPA*NOCTPB)
      DO 100 IATP = 1, NOCTPA
         DO 90 IBTP = 1, NOCTPB
*. is this combination allowed in any of the GAS spaces included
           INCLUDE = 0
           DO JJCMBSPC = 1, LCMBSPC
             JCMBSPC = ICMBSPC(JJCMBSPC)
             IEL = 0
             IAMOKAY = 1
             DO IGAS = 1, NGAS
               IEL = IEL
     &             + NELFTP(IOCA(IGAS,IATP))+NELFTP(IOCB(IGAS,IBTP))
               IF(IEL.LT.MNMXOC(IGAS,1,JCMBSPC).OR.
     &            IEL.GT.MNMXOC(IGAS,2,JCMBSPC))
     &         IAMOKAY = 0
             END DO 
             IF(IAMOKAY.EQ.1) INCLUDE = 1
           END DO
* 
           IF(I_RE_MS2_SPACE.NE.0) THEN
*. Spin projection after space I_RE_MS2_SPACE:
             MS2_INTERM = 0
             DO IGAS = 1, I_RE_MS2_SPACE
               MS2_INTERM = MS2_INTERM +
     &         NELFTP(IOCA(IGAS,IATP))-NELFTP(IOCB(IGAS,IBTP))
             END DO
             IF(MS2_INTERM.NE.I_RE_MS2_VALUE) THEN
               INCLUDE = 0
             END IF
           END IF
*
           IF(I_CHECK_ENSGSOCC.EQ.1) THEN
*. Check that the number of electrons in the ensemble gas space 
* is within limit.
            DO IGAS = 1, NGAS
              IGSOCC(IGAS) = NELFTP(IOCA(IGAS,IATP)) +
     &                       NELFTP(IOCB(IGAS,IBTP))
            END DO
            CALL CHECK_IS_OCC_IN_ENGSOCC(IGSOCC,JCMBSPC,IM_IN)
            IF(IM_IN.EQ.0) INCLUDE = 0
C?          WRITE(6,*) ' IM_IN, INCLUDE = ', IM_IN, INCLUDE
           END IF
*
           IF(INCLUDE.EQ.1) THEN 
*. Congratulations , you are allowed
              IOCOC(IATP,IBTP) = 1
          END IF
   90   CONTINUE
  100 CONTINUE
*
      IF ( NTEST .GE. 100 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' Matrix giving allowed combinations of types '
        WRITE(6,*)
        CALL IWRTMA(IOCOC,NOCTPA,NOCTPB,NOCTPA,NOCTPB)
      END IF
*
      RETURN
      END
      SUBROUTINE ICPMT2(AIN,AOUT,NINR,NINC,NOUTR,NOUTC,IZERO)
*
* Copy INTEGER matrix AIN to AOUT . Dimensions can differ
*
* If IZERO .ne. 0 , AOUT is zeroed  first
      IMPLICIT REAL*8           (A-H,O-Z)
*. Input
      INTEGER AIN(NINR,NINC)
*. Output
      INTEGER AOUT(NOUTR,NOUTC)
*
      IF(IZERO.NE.0) CALL ISETVC(AOUT,0,NOUTR*NOUTC)
      DO 100 J = 1, NINC
       CALL ICOPVE(AIN(1,J),AOUT(1,J),NINR)
  100 CONTINUE
*
      RETURN
      END
      FUNCTION IELSUM(IVEC,NELMNT)
*
* Sum elements of integer vector IVEC
*
      DIMENSION IVEC(*)
*
      ISUM = 0
      DO 100 IELMNT = 1, NELMNT
        ISUM = ISUM + IVEC(IELMNT)
  100 CONTINUE
*
      IELSUM = ISUM
*
      RETURN
      END
      FUNCTION IFNDNM(IA,NDIM,IVAL)
*
* Find first element in integer array IA, that has value IVAL
*
      INTEGER IA(*)
*
      IELMNT = 0
      DO 100 I = 1, NDIM
        IF(IA(I).EQ.IVAL) THEN
          IELMNT = I
          GOTO 101
        END IF
  100 CONTINUE
  101 CONTINUE
*
      IFNDNM = IELMNT
      RETURN
      END
      FUNCTION IFREQ(IVEC,IVAL,NDIM)
*
* Number of times IVAL occurs in IVEC
*
      DIMENSION IVEC(*)
*
      NTIME = 0
      DO 100 I = 1, NDIM
        IF(IVEC(I).EQ.IVAL) NTIME = NTIME + 1
  100 CONTINUE
*
      IFREQ = NTIME
*
      RETURN
      END
      FUNCTION IFRMR(int_mb,IROFF,IELMNT)
*
* An integer array is stored in real array WORK,
* starting from WORK(IROFF). Obtain element
* IELMNT of this array
*
      INTEGER int_mb(*)
*
      INCLUDE 'irat.inc'
*. offset when work is integer array
      IIOFF = 1 + IRAT * (IROFF-1)
      IFRMR = int_mb(IIOFF-1+IELMNT)
*
      RETURN
      END
      FUNCTION IMNMX(IVEC,NDIM,MINMAX)
*
*     Find smallest (MINMAX=1) or largest (MINMAX=2)
*     absolute value of elements in integer vector IVEC
*
      DIMENSION IVEC(1)
*
      IX = 0
      IF(NDIM.GT.0) THEN
        IX = -1
        IF(MINMAX.EQ.1) THEN
          IX=ABS(IVEC(1))
          DO I=2,NDIM
            IX=MIN(IX,ABS(IVEC(I)))
          END DO
        END IF
*
        IF(MINMAX.EQ.2) THEN
          IX=ABS(IVEC(1))
          DO I=2,NDIM
            IX=MAX(IX,ABS(IVEC(I)))
          END DO
        END IF
*
      ELSE IF(NDIM.EQ.0) THEN
*. No components: set to zero and write a warning
        IX = 0
C       WRITE(6,*) ' Min/Max taken zero length vector set to zero'
      END IF
*
      IMNMX = IX
*
      RETURN
      END
      SUBROUTINE IMNXVC(IVEC,NDIM,MXMN,IVAL,IPLACE)
C
C MXMN = 1: FIND LARGEST ELEMENT IN IVEC
C MXMN = 2: FIND SMALLEST ELEMENT IN IVEC
C
C RESULTING VALUE: IVAL
C PLACE OF RESULTING VALUE: IPLACE
C
      DIMENSION IVEC(*)
C
      IVAL = IVEC(1)
      IPLACE = 1
      IF( MXMN .EQ. 1 ) THEN
        DO 100 I = 2, NDIM
          IF(IVEC(I) .GE. IVAL ) THEN
            IVAL = IVEC(I)
            IPLACE = I
          END IF
  100   CONTINUE
      ELSE IF ( MXMN .EQ. 2 ) THEN
        DO 200 I = 2, NDIM
          IF(IVEC(I) .LE. IVAL ) THEN
            IVAL = IVEC(I)
            IPLACE = I
          END IF
  200   CONTINUE
      END IF
C
      NTEST = 01
      IF( NTEST .NE. 0 )
     &WRITE(6,*) ' MXMN IVAL IPLACE ' ,MXMN,IVAL,IPLACE
C
      RETURN
      END
      SUBROUTINE INCOOS(IDC,IBLTP,NOOS,NOCTPA,NOCTPB,ISTSM,ISTTA,ISTTB,
     &                  NSMST,IENSM,IENTA,IENTB,IACOOS,MXLNG,IFINI,
     &                  NBLOCK,INCFST,IOCOC)
*
* Obtain Number of OOS blocks that can be included
* IN MXLNG word starting from block after ISTSM,ISTTA,ISTTB
* Activated blocks are given in IACOOS
* Last activated block is (IENSM,IENTA,IENTB)
* If all blocks have been accessed IFINI is returned as 1
* Diagonal blocks are expanded
*
* Jeppe Olsen, Winter of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      INTEGER NOOS(NOCTPA,NOCTPB,NSMST)
      INTEGER IOCOC(NOCTPA,NOCTPB)
C-May 7
      INTEGER IBLTP(*)
C-May 7
*.Output
      INTEGER IACOOS(NOCTPA,NOCTPB,NSMST)
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*)
        WRITE(6,*) ' =================='
        WRITE(6,*) ' INCOOS in action  '
        WRITE(6,*) ' =================='
        WRITE(6,*)
        WRITE(6,*) ' NOOS(NOCTPA,NOCTPB,NSMST) array (input) '
        WRITE(6,*)
        DO ISMST = 1, NSMST
         WRITE(6,*) ' ISMST = ', ISMST
         CALL IWRTMA(NOOS(1,1,ISMST),NOCTPA,NOCTPB,NOCTPA,NOCTPB)
        END DO
      END IF
*
      IPA = 0
      IPB = 0
      IPSM = 0
*
*.Initialize
      CALL ISETVC(IACOOS,0,NOCTPA*NOCTPB*NSMST)
      IFRST = 1
      ISM = ISTSM
      IA = ISTTA
      IB = ISTTB
      LENGTH = 0
      NBLOCK = 0
      IENSM = ISTSM
      IENTA = ISTTA
      IENTB = ISTTB
      IFINI = 0
      IF(INCFST.EQ.1) GOTO 999
 1000 CONTINUE
*.Next block
      IPA = IA
      IPB = IB
      IPSM = ISM
*
      IF(IB.LT.NOCTPB) THEN
        IB = IB + 1
      ELSE
        IB = 1
        IF(IA.LT.NOCTPA) THEN
          IA = IA+ 1
        ELSE
          IA = 1
          IF(ISM.LT.NSMST) THEN
            ISM = ISM + 1
          ELSE
            IFINI = 1
          END IF
        END IF
      END IF
      IF(IFINI.EQ.1) GOTO 1001
*. Should this block be included
  999 CONTINUE
      IF(IDC.NE.1.AND.IBLTP(ISM).EQ.0) GOTO 1000
      IF(IDC.NE.1.AND.IBLTP(ISM).EQ.2.AND.IA.LT.IB) GOTO 1000
      IF(IOCOC(IA,IB).EQ.0) GOTO 1000
C?    write(6,*) ' INCOOS IDC IBLTP ', IDC,IBLTP(ISM)
*. can this block be included
      LBLOCK = NOOS(IA,IB,ISM)
C?    write(6,*) ' IA IB ISM LBLOCK ', IA,IB,ISM,LBLOCK
      IF(LENGTH+LBLOCK.LE.MXLNG) THEN
        NBLOCK = NBLOCK + 1
        LENGTH = LENGTH + LBLOCK
        IACOOS(IA,IB,ISM) = 1
        IF(NBLOCK.EQ.1) THEN
          ISTTA = IA
          ISTTB = IB
          ISTSM = ISM
         END IF
        GOTO 1000
      ELSE
        IA = IPA
        IB = IPB
        ISM = IPSM
      END IF
 1001 CONTINUE
*
      IENSM = ISM
      IENTA = IA
      IENTB = IB
      IF(IFINI.EQ.0.AND.NBLOCK.EQ.0) THEN
        WRITE(6,*) ' Not enough scratch space to include a single Block'
        WRITE(6,*) ' Since I cannot procede I will stop '
        WRITE(6,*) ' Insufficient buffer detected in INCOOS '
        WRITE(6,*) ' Alter RAS space of raise Buffer from ', MXLNG
        CALL MEMCHK
        STOP 11
      END IF
*
      IF(NTEST.NE.0) THEN
        WRITE(6,*) 'Output from INCOOS '
        WRITE(6,*) '==================='
        WRITE(6,*)
     &  ' Length and number of included blocks ',LENGTH,NBLOCK
      END IF
      IF(NTEST.GE.2) THEN
        DO 100 ISM = ISTSM,IENSM
          WRITE(6,*) ' Active blocks of symmetry ',ISM
          CALL IWRTMA(IACOOS(1,1,ISM),NOCTPA,NOCTPB,NOCTPA,NOCTPB)
  100   CONTINUE
        IF(IFINI.EQ.1) WRITE(6,*) ' No new blocks '
      END IF
*
      RETURN
      END
      SUBROUTINE INPCTL(LUIN,LUOUT)
*
* Read and check input for LUCIA
*
*.0: Look at inter-program control files
      CALL INTERACT
*.1: Position input file after  line containing &LUCIA &END
C      CALL SLASK
*.2: read input in
      CALL READIN(LUIN,LUOUT)
      RETURN
      END
c
      SUBROUTINE INTERACT
c
c     read automatically generated input to LUCIA
c
      INCLUDE 'implicit.inc'
      INCLUDE 'symrdc.inc'

      LOGICAL LEXIST
      CHARACTER FILNAM*10,BLABLA*80

*. defaults
      SYMRED = .FALSE.

      FILNAM = 'LUCIA.IAC'
      INQUIRE(FILE=FILNAM,EXIST=LEXIST)
      IF (LEXIST) THEN
        LUIAC = IOPEN_NFS(FILNAM)
        REWIND LUIAC
        DO                      ! loop over lines in file
          READ(LUIAC,'(A12)',END=110,ERR=100) BLABLA
          IF (BLABLA(1:1).EQ.'!') CYCLE
          IF (BLABLA(1:12).EQ.'*SYMMETRY RE') THEN
*.a) symmetry reduction (relative to LUCIAs input file)
            SYMRED = .TRUE.
            READ(LUIAC,'(A14,I3)') BLABLA,NIRREP_OLD
            READ(LUIAC,'(A14,I3)') BLABLA,NIRREP_NEW
            READ(LUIAC,'(A)') BLABLA
            NQUOT = NIRREP_OLD/NIRREP_NEW
            IF (NQUOT.NE.1.AND.NQUOT.NE.2.AND.
     &           NQUOT.NE.4.AND.NQUOT.NE.8) THEN
              WRITE(6,*) 'strange input from LUCIA.IAC; I quit ...'
              WRITE(6,*) 'NIRREP_OLD = ',NIRREP_OLD
              WRITE(6,*) 'NIRREP_NEW = ',NIRREP_NEW
              STOP 'I quit on LUCIA.SBD'
            END IF
            DO ILINE = 1, NIRREP_NEW
              READ(LUIAC,*) (IRMAP((ILINE-1)*NQUOT+IDX), IDX = 1,NQUOT)
            END DO
            ! test whether really symmetry reduction occurs:
            IF (NIRREP_NEW.EQ.NIRREP_OLD) THEN
              ! ... maybe not ...
              SYMRED = .FALSE.
              ! ... unless we have to resort for some reason ...
              DO II = 1, NIRREP_NEW
                IF (IRMAP(II).NE.II) THEN
                  ! ok, we did it ....
                  SYMRED = .TRUE.
                  EXIT
                END IF
              END DO
            END IF
          ELSE IF(BLABLA(1:10).EQ.'*DO ENERGY') THEN
            WRITE(6,*) 'found DO ENERGY, but do not care ...'
          ELSE IF(BLABLA(1:12).EQ.'*DO GRADIENT') THEN
            WRITE(6,*) 'found DO GRADIENT, but do not care ...'
          ELSE IF(BLABLA(1:4).EQ.'*END') THEN
            GOTO 110
          ELSE
            WRITE(6,*) 'found unexpected entry on LUCIA.IAC:'
            WRITE(6,*) BLABLA(1:LEN_TRIM(BLABLA))
            WRITE(6,*) 'ignoring this line and continuing ....'
          END IF
        END DO
 100    CONTINUE
        WRITE(6,*)
     &       'Error reading from LUCIA.IAC or premature end of file'
        STOP 'Error reading from LUCIA.IAC'
 110    CONTINUE
        ! we delete the interaction file, as it might else lead to
        ! misunderstandings
        CALL RELUNIT(LUIAC,'delete')

      END IF

      END 
C     GASDIAT(WORK(KVEC1),LUDIA,ECOREP,ICISTR,I12,
C    &               WORK(KLCBLTP),NBLOCK,WORK(KLCIBT))
      SUBROUTINE GASDIAT(DIAG,LUDIA,ECORE,ICISTR,I12,
     &           IBLTP,NBLOCK,IBLKFO,IEXP_PH)
*
* CI diagonal in SD basis for state with symmetry ISM in internal
* space ISPC
*
* GAS version, Winter of 95
*
* Driven by table of TTS blocks, May97
* Small change of memory allocation, Aug. 03
* PH version added, June 10 ( about 10 years after intro of ph..)
*
* IEXP_PH = 1: Use version with explicit use of ph simplifications
*
c      IMPLICIT REAL*8(A-H,O-Z)
* =====
*.Input
* =====
*
*./ORBINP/: NACOB used
*
c      INCLUDE 'mxpdim.inc'
#include "mafdecls.fh"
      INCLUDE 'wrkspc.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'glbbas.inc'
*
      DIMENSION IBLTP(*)
      DIMENSION IBLKFO(8,NBLOCK)
*
* ======
*.Output
* ======
      DIMENSION DIAG(*)
*
      CALL QENTER('CIDIA')
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRDIA)
*
** Specifications of internal space
*
      IATP = 1
      IBTP = 2
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
* 
*. Offsets for alpha and beta supergroups
      IOCTPA = IBSPGPFTP(IATP)
      IOCTPB = IBSPGPFTP(IBTP)
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' ================'
        WRITE(6,*) ' GASDIA speaking '
        WRITE(6,*) ' ================'
        WRITE(6,*) ' IATP IBTP NAEL NBEL = ',IATP,IBTP,NAEL,NBEL
        write(6,*) ' NOCTPA NOCTPB = ', NOCTPA,NOCTPB
        write(6,*) ' IOCTPA IOCTPB = ', IOCTPA,IOCTPB
        WRITE(6,*) ' IEXP_PH = ', IEXP_PH
        WRITE(6,*) ' Output to unit: ', LUDIA
      END IF
*
**. Local memory
*
      IDUM = 0
      CALL MEMMAN(IDUM,  IDUM,    'MARK  ',IDUM,'GASDIA')
      CALL MEMMAN(KLJ   ,NTOOB**2,'ADDL  ',2,'KLJ   ')
      CALL MEMMAN(KLK   ,NTOOB**2,'ADDL  ',2,'KLK   ')
      CALL MEMMAN(KLSCR2,2*NTOOB**2,'ADDL  ',2,'KLSC2 ')
      CALL MEMMAN(KLXA  ,NTOOB,   'ADDL  ',2,'KLXA  ')
      CALL MEMMAN(KLXB  ,NTOOB,   'ADDL  ',2,'KLXB  ')
      CALL MEMMAN(KLSCR ,2*NTOOB, 'ADDL  ',2,'KLSCR ')
      CALL MEMMAN(KLISCR,NTOOB,   'ADDL  ',1,'KLISCR ')
      CALL MEMMAN(KLH1D ,NTOOB,   'ADDL  ',2,'KLH1D ')
*. Space for blocks of strings
      MAXA = IMNMX(int_mb(KNSTSO(IATP)),NSMST*NOCTPA,2)
      MAXB = IMNMX(int_mb(KNSTSO(IBTP)),NSMST*NOCTPB,2)
      CALL MEMMAN(KLRJKA,MAXA,'ADDL  ',2,'KLRJKA')
      CALL MEMMAN(KLASTR,MAXA*NAEL,'ADDL  ',1,'KLASTR')
      CALL MEMMAN(KLBSTR,MAXB*NBEL,'ADDL  ',1,'KLBSTR')
*. One block of strings in phformat
      IF(IEXP_PH.EQ.1) THEN
        CALL MEMMAN(KLPHSTR,MAX_STR_PHOC_BLK,'ADDL  ',1,'KLPHOC')
      END IF
      CALL MEMMAN(KLRJKA,MAXA,'ADDL  ',2,'KLRJKA')
*
** Info on block structure of internal state
*
*
**. Diagonal of one-body integrals and coulomb and exchange integrals
*   One-body integrals stored in KINT1 are used
*
      CALL GT1DIA(dbl_mb(KLH1D))
      CALL GTJK(dbl_mb(KLJ),dbl_mb(KLK),NTOOB,dbl_mb(KLSCR2),IREOTS)
*
      IF( LUDIA .GT. 0 ) CALL REWINO(LUDIA)
      IF(IEXP_PH.EQ.0) THEN
*. Good old version where PH is not used for diagonal
        CALL GASDIAS(NAEL,int_mb(KLASTR),NBEL,int_mb(KLBSTR),
     &       NACOB,DIAG,NSMST,dbl_mb(KLH1D),
     &       dbl_mb(KLXA),dbl_mb(KLXB),dbl_mb(KLSCR),dbl_mb(KLJ),
     &       dbl_mb(KLK),int_mb(KNSTSO(IATP)),int_mb(KNSTSO(IBTP)),
     &       LUDIA,ECORE,PLSIGN,PSSIGN,IPRDIA,NTOOB,ICISTR,
     &       dbl_mb(KLRJKA),I12,IBLTP,NBLOCK,IBLKFO)
      ELSE
*. use also ph in diagonal construction
        CALL GASDIAS_PH(NAEL,int_mb(KLASTR),NBEL,int_mb(KLBSTR),
     &       NACOB,DIAG,NSMST,dbl_mb(KLH1D),
     &       dbl_mb(KLXA),dbl_mb(KLXB),dbl_mb(KLSCR),dbl_mb(KLJ),
     &       dbl_mb(KLK),int_mb(KNSTSO(IATP)),int_mb(KNSTSO(IBTP)),
     &       LUDIA,ECORE,PLSIGN,PSSIGN,IPRDIA,NTOOB,ICISTR,
     &       dbl_mb(KLRJKA),I12,IBLTP,NBLOCK,IBLKFO,IPHGAS,ITPFTO,
     &       int_mb(KLPHSTR),NPHELFSPGP,IOCTPA,IOCTPB,NINOB,
     &       int_mb(KLISCR))
      END IF

*.Flush local memory
      CALL MEMMAN(IDUM,  IDUM,    'FLUSM ',IDUM,'GASDIA')
      CALL QEXIT('CIDIA')
*
      RETURN
      END
 
      SUBROUTINE INTDIM(IPRNT)
*
* Number of integrals and storage mode
*
      IMPLICIT REAL*8(A-H,O-Z)
*
* =====
*.Input
* =====
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'csmprd.inc'
c*.CSMPRD
c      INTEGER ADASX,ASXAD,ADSXA,SXSXDX,SXDXSX
c      COMMON/CSMPRD/ADASX(MXPOBS,MXPOBS),ASXAD(MXPOBS,2*MXPOBS),
c     &              ADSXA(MXPOBS,2*MXPOBS),
c     &              SXSXDX(2*MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
*
* =======
*. Output
* =======
*
      INCLUDE 'cintfo.inc'
*
*.1: Number of one-electron integrals
*
      NINT1 =  NSXFSM(NSMOB,MXPOBS,NTOOBS,NTOOBS,ITSSX,ADSXA,1,IPRNT)
*
*.2: Number of two-electron integrals
*
      IF(PNTGRP.EQ.1.AND.IDO_LIPKIN.EQ.0) THEN
*. Full eightfold symmetry can be used
        I12S = 1
        I34S = 1
        I1234S = 1
      ELSE
*. Only symmetry between 12 and 34
        I12S = 0
        I34S = 0
        I1234S = 1
      END IF
      NINT2 = NDXFSM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,
     &                NTOOBS,ITSDX,ADSXA,SXDXSX,I12S,I34S,I1234S,
     &                IPRNT )
*. Number of integrals without complex conjugation symmetry 
*. ( used for unrestricted a/b orbitals)
      I12 = 1
      I34 = 1
      I1234 = 0
      NINT2_NO12SYM = NDXFSM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,
     &                NTOOBS,ITSDX,ADSXA,SXDXSX,I12,I34,I1234,
     &                IPRNT )

*. Number of integrals without complex conjugation symmetry 
*. ( used for T1 transformed Hamiltonian) 
      I12 = 0
      I34 = 0
      I1234 = 1
      NINT2_NO_CCSYM = NDXFSM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,
     &                 NTOOBS,ITSDX,ADSXA,SXDXSX,I12,I34,I1234,
     &                 IPRNT )
*. Number of integrals without complex conjugation symmetry 
*. and without symmetry between particle one and two
*. ( used for alpha-beta part of similarity transformed H)
      I12 = 0
      I34 = 0
      I1234 = 0
      NINT2_NO_CCSYM_NO12SYM 
     &      = NDXFSM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,
     &        NTOOBS,ITSDX,ADSXA,SXDXSX,I12,I34,I1234,
     &        IPRNT )
      
       IF(ISIMTRH.EQ.1) THEN
         IF(IREFTYP.NE.2) THEN
         WRITE(6,*) 
     &   ' Number of two-electron integrals in exp(-T1)Hexp(T1) ',
     &     NINT2_NO_CCSYM
         ELSE
         WRITE(6,*) 
     &   ' Number of two-electron integrals in exp(-T1)Hexp(T1) ',
     &     2*NINT2_NO_CCSYM + NINT2_NO_CCSYM_NO12SYM
         END IF
       END IF
*. Number of symmetry blocks of one- and two-electron integrals
      NBINT1 = NSMOB
      NBINT2 = NSMOB ** 3
      RETURN
      END
      SUBROUTINE Z_TYP_EI_LISTS
*
* Set up types of integral lists, and store in 
* NE2LIST,IE2LIST_NARR(MXP2EIARR),IE2LIST_IARR(4,MXP2EIARR,MXP2EIARR),
* IE1_CCSM_G, IE2_CCSM_G
* in CINTFO
*
*. Jeppe Olsen, April 2011, Lucia growing up
*               July 2011, IH1_12_G added 
*
      INCLUDE 'wrkspc.inc'
      INCLUDE 'cintfo.inc'
*. Local 
      CHARACTER*1 OGC(2)
*
* A list of integral may consists of one or several integral arrays
* where each array is defined by a given type of orbitals (occupied or
* general) in each index
*
* Type of orbitals is flagged by an index IOG being 1(occ) or 2(gen)
*
*
      NTEST = 100
* ====================
* Arrays of integrals
* ====================
*
* Array 1: zero general indeces
*
      IE2ARR = 1
      I12S_G(IE2ARR) = 1
      I34S_G(IE2ARR) = 1
      I1234S_G(IE2ARR) = 1
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 1
      INT2ARR_G(3,IE2ARR) = 1
      INT2ARR_G(4,IE2ARR) = 1
*
* Array 2: one general indeces: (OO!OG)
*
      IE2ARR = 2
      I12S_G(IE2ARR) = 1
      I34S_G(IE2ARR) = 0
      I1234S_G(IE2ARR) = 0
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 1
      INT2ARR_G(3,IE2ARR) = 1
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 3: two general indeces: (OO!GG)
*
      IE2ARR = 3
      I12S_G(IE2ARR) = 1
      I34S_G(IE2ARR) = 1
      I1234S_G(IE2ARR) = 0
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 1
      INT2ARR_G(3,IE2ARR) = 2
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 4: two general indeces: (OG!OG)
*
      IE2ARR = 4
      I12S_G(IE2ARR) = 0
      I34S_G(IE2ARR) = 0
      I1234S_G(IE2ARR) = 1
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 2
      INT2ARR_G(3,IE2ARR) = 1
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 5: three general indeces: (OG!GG)
*
      IE2ARR = 5
      I12S_G(IE2ARR) = 0
      I34S_G(IE2ARR) = 1
      I1234S_G(IE2ARR) = 0
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 2
      INT2ARR_G(3,IE2ARR) = 2
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 6: Four general indeces: (GG!GG)
*
      IE2ARR = 6
      I12S_G(IE2ARR) = 1
      I34S_G(IE2ARR) = 1
      I1234S_G(IE2ARR) = 1
      INT2ARR_G(1,IE2ARR) = 2
      INT2ARR_G(2,IE2ARR) = 2
      INT2ARR_G(3,IE2ARR) = 2
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 7: Four general indices (GG!GG) and symmetry for biortogonal expansion
*
      IE2ARR = 7
      I12S_G(IE2ARR) = 0
      I34S_G(IE2ARR) = 0
      I1234S_G(IE2ARR) = 1
      INT2ARR_G(1,IE2ARR) = 2
      INT2ARR_G(2,IE2ARR) = 2
      INT2ARR_G(3,IE2ARR) = 2
      INT2ARR_G(4,IE2ARR) = 2
*
* Array 8: One general index and symmetry for bioorthogonal expansion
*
      IE2ARR = 8
      I12S_G(IE2ARR) = 0
      I34S_G(IE2ARR) = 0
      I1234S_G(IE2ARR) = 0
      INT2ARR_G(1,IE2ARR) = 1
      INT2ARR_G(2,IE2ARR) = 1
      INT2ARR_G(3,IE2ARR) = 1
      INT2ARR_G(4,IE2ARR) = 2
*
      NE2ARR = 8
*
* ====================
*. And now the lists
* ====================
*
      NE2LIST = 6
*
* List 1: Zero free indeces
* 
      IE2LIST = 1
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 1
      IE2LIST_I(1) = 1
      IE2LIST_0G = IE2LIST
      IE1_CCSM_G(IE2LIST) = 1
      IE2_CCSM_G(IE2LIST) = 1
*
* List 2: one free index
*
      IE2LIST = 2
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 2
      IE2LIST_I(2) = 2
      IE2LIST_1G = IE2LIST
      IE1_CCSM_G(IE2LIST) = 1
      IE2_CCSM_G(IE2LIST) = 1
*
* List 3: two free index
*
      IE2LIST = 3
      IE2LIST_N(IE2LIST) = 2
      IE2LIST_IB(IE2LIST) = 3
      IE2LIST_I(3) = 3
      IE2LIST_I(4) = 4
      IE2LIST_2G = IE2LIST
      IE1_CCSM_G(IE2LIST) = 1
      IE2_CCSM_G(IE2LIST) = 1
*
* List 4: three free index
*
      IE2LIST = 4
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 5
      IE2LIST_I(5) = 5
      IE2LIST_3G = IE2LIST
      IE1_CCSM_G(IE2LIST) = 1
      IE2_CCSM_G(IE2LIST) = 1
*
* List 5: Four free index
*
      IE2LIST = 5
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 6
      IE2LIST_I(6) = 6
      IE1_CCSM_G(IE2LIST) = 1
      IE2_CCSM_G(IE2LIST) = 1
      IE2LIST_4G = IE2LIST
      IE2LIST_FULL = IE2LIST_4G
*
* List 6: Four free indeces and permutational symmetry for bioorthogonal expansion
*
      IE2LIST = 6
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 7
      IE2LIST_I(7) = 7
      IE1_CCSM_G(IE2LIST) = 0
      IE2_CCSM_G(IE2LIST) = 0
      IE2LIST_FULL_BIO = IE2LIST
*
* List 7: One free indeces and permutational symmetry for bioorthogonal expansion
*
      IE2LIST = 7
      IE2LIST_N(IE2LIST) = 1
      IE2LIST_IB(IE2LIST) = 8
      IE2LIST_I(8) = 8
      IE1_CCSM_G(IE2LIST) = 0
      IE2_CCSM_G(IE2LIST) = 0
      IE2LIST_1G_BIO = IE2LIST

      IF(NTEST.GE.100) THEN

        WRITE(6,*) 
     & ' Information on defined lists of two-electron-integrals'
        WRITE(6,*) 
     & ' ======================================================'
*
        OGC(1:1) = 'O'
        OGC(2:2) = 'G'
        WRITE(6,*) '( I J ! K L) I12SM I34SM I1234SM '
        DO IARR = 1, NE2ARR
          WRITE(6,'(7A2,I3,3X,I3,3X,I3,3X)')
     &    ' (', OGC(INT2ARR_G(1,IARR)),
     &          OGC(INT2ARR_G(2,IARR)),
     &    ' !', OGC(INT2ARR_G(3,IARR)),
     &          OGC(INT2ARR_G(4,IARR)),') ',
     &     I12S_G(IARR),I34S_G(IARR),I1234S_G(IARR)
        END DO
*
        WRITE(6,*) ' Integral lists => integral arrays '
        DO ILIST = 1, NE2LIST
          IB = IE2LIST_IB(ILIST)
          N  = IE2LIST_N(ILIST)
          WRITE(6,'(A,I2,A,10I2)') ' Integral list ', ILIST,
     &    ' contains arrays ',  (IE2LIST_I(I),I=IB,IB-1+N)
        END DO
      END IF
*
      RETURN
      END
*
      SUBROUTINE INTDIM_G(IPRNT)
*
* Obtain number of transformed two-electron integrals
* for the various integral lists and set up various 
* arrays defining integral arrays
*
* Jeppe Olsen, April 2011, still the LUCIA growing up campaign
*
* The results are stored in N2INT_G which resides in CINTFO
*
      INCLUDE 'wrkspc.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'csmprd.inc'
      INCLUDE 'orbinp.inc'
*. Local scratch
      INTEGER NOCPSM_LO(MXPOBS), NOCPSM_L(MXPOBS,4)
      INTEGER NOCPTSM_LO(MXPOBS*(7+MXPR4T))
      INTEGER ISUBTP(2), IPN_L(8,8,8),ISM_L(8,8,8)
      
*
* Done with two types of occupied: 1 => Occupied = active
*                                  2 => Occupied = active + inactiv
       DO IOCTP = 1, 2
*. Number of occupied per symmetry
        IF(IOCTP.EQ.1) THEN
          NSUBTP = 1
          ISUBTP(1) = NGAS
          CALL CSUB_FROM_C(XDUM,XDUM,NOCPSM_LO,NOCPTSM_LO,
     &         NSUBTP,ISUBTP,1)
C              CSUB_FROM_C(C,CSUB,LENSUBS,NSUBTP,ISUBTP,IONLY_DIM)
        ELSE
          NSUBTP = 2
          ISUBTP(1) = 0
          ISUBTP(2) = NGAS
          CALL CSUB_FROM_C(XDUM,XDUM,NOCPSM_LO,NOCPTSM_LO,
     &          NSUBTP,ISUBTP,1)
        END IF 
        DO IARR = 1, NE2ARR
          DO INDEX = 1, 4
            IF(INT2ARR_G(INDEX,IARR).EQ.1) THEN
              CALL ICOPVE(NOCPSM_LO,NOCPSM_L(1,INDEX),NSMOB)
            ELSE 
              CALL ICOPVE(NTOOBS,NOCPSM_L(1,INDEX),NSMOB)
            END IF
          END DO
          I12S_L = I12S_G(IARR)
          I34S_L = I34S_G(IARR)
          I1234S_L = I1234S_G(IARR)
          DO ISM = 1, NSMOB
             CALL PNT4DM(NSMOB,NSMSX,MXPOBS,
     &       NOCPSM_L(1,1),NOCPSM_L(1,2),NOCPSM_L(1,3),
     &       NOCPSM_L(1,4),ISM,ADSXA,SXDXSX,I12S_L,I34S_L,I1234S_L,
     &       IPN_L, ISM_L,ADASX,NINT4D)
             N2INTARR_G(ISM,IARR,IOCTP) = NINT4D
          END DO ! End of loop over symmetries
        END DO !End of loop over types of integral arrays
      END DO !End of loop over the two types of occupied orbitals
*
*. And then the dimension of the integral lists.
*
      DO IOCTP = 1, 2
        DO ILIST = 1, NE2LIST
          DO ISM = 1, NSMOB
            N = IE2LIST_N(ILIST)
            IB = IE2LIST_IB(ILIST)
            LENGTH = 0
            DO IARR = IB,IB-1+N
              IIARR = IE2LIST_I(IARR)
              LENGTH = LENGTH + N2INTARR_G(ISM,IIARR,IOCTP)
            END DO
            N2INTLIS_G(ISM,ILIST,IOCTP) = LENGTH
          END DO
        END DO
      END DO
*
      NTEST = 000
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Length of integralarrays '
        WRITE(6,*) ' ========================='
        WRITE(6,*) 
        WRITE(6,*) ' Occupied = active:  '
        WRITE(6,*)
        WRITE(6,*) ' row = sym, column = type of array '
        CALL IWRTMA(N2INTARR_G(1,1,1),
     &              NSMOB,NE2ARR,MXPOBS,MXP2EIARR)
        WRITE(6,*) 
        WRITE(6,*) ' Occupied = active + occupied:  '
        WRITE(6,*)
        WRITE(6,*) ' row = sym, column = type of array '
        CALL IWRTMA(N2INTARR_G(1,1,2),
     &              NSMOB,NE2ARR,MXPOBS,MXP2EIARR)
        WRITE(6,*)
        WRITE(6,*) ' Length of integrallists '
        WRITE(6,*) ' ========================'
        WRITE(6,*) 
        WRITE(6,*) ' Occupied = active:  '
        WRITE(6,*)
        WRITE(6,*) ' row = sym, column = type of array '
        CALL IWRTMA(N2INTLIS_G(1,1,1),
     &              NSMOB,NE2LIST,MXPOBS,MXP2EIARR)
        WRITE(6,*) 
        WRITE(6,*) ' Occupied = active + occupied:  '
        WRITE(6,*)
        WRITE(6,*) ' row = sym, column = type of array '
        CALL IWRTMA(N2INTLIS_G(1,1,2),
     &              NSMOB,NE2LIST,MXPOBS,MXP2EIARR)
      END IF
*
      RETURN
      END
      SUBROUTINE INTIM(IPRNT)
*
* Interface to external integrals
*
* If NOINT .ne. 0, only pointers are constructed
* Jeppe Olsen, Winter of 1991
*
* Version: Fall 97
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cecore.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'intform.inc'
      CALL MEMCHK
*. Integrals will be saved in KINT_2EMO, so
      KINT2 = KINT_2EMO
*. And have normal permutational symmetry, so
      IH1FORM = 1
      IH2FORM = 2
*
*.: Pointers for symmetry blocks of integrals
*
      CALL INTPNT(WORK(KPINT1),WORK(KLSM1),
     &            WORK(KPINT2),WORK(KLSM2))
*. Pointer for orbital indeces for symmetry blocked matrices
      CALL ORBINH1(WORK(KINH1),WORK(KINH1_NOCCSYM),NTOOBS,NTOOB,NSMOB)
*
*  ----------------------------------------------------------------
*. Read integrals in according to the specification of environment
*  ----------------------------------------------------------------
*
      IF(NOINT.EQ.0) THEN
*
       IF(IDOQD.EQ.1) THEN
*
* ===================================================
*. Obtain integrals from qdot- transformation program
* ===================================================
*
        WRITE(6,*) ' GET_QD_INTS will be called '
        ZERO = 0.0D0
        CALL SETVEC(WORK(KINT1), ZERO, NINT1)
        CALL SETVEC(WORK(KINT2), ZERO, NINT2)
        CALL GET_QD_INTS(WORK(KINT1),WORK(KINT2),WORK(KPINT1))
        INTIMP = 1
       ELSE IF(IDOQD.EQ.0.AND.INTIMP.EQ.1) THEN
*
*  ==============
* . Molcas format
*  ==============
*
        WRITE(6,*) ' Integrals imported from MOLCAS files'
*.Initialize buffers, open
        CALL MKLUNDIO
*. Load one-electron integrals
        CALL GETH0(WORK(KINT1))
*. And two-electron integrals if desired
        IF(INCORE.EQ.1.AND.ISVMEM.EQ.0) THEN
          IF(ITRA_ROUTE.EQ.1) THEN
            CALL INTIMM(WORK(KINT2),NSMOB)
          ELSE
            CALL INTIMM(WORK(KINT2),NSMOB)
            WRITE(6,*) ' MOLCAS route has not been programmed for '
            WRITE(6,*) ' new ordering of integrals '
cGLM            STOP ' MOLCAS input not programmed for ITRA_ROUTE = 2'
          END IF
        END IF
       ELSE IF (INTIMP.EQ.3) THEN
*
* ===================================================
* Formatted input of symmetry non-vanishing integrals
* ===================================================
*
        WRITE(6,*) ' Integrals imported formatted (E22.15) '
*.1: One-electron integrals
        REWIND LU2INT
        READ(LU2INT,'(E22.15)') (WORK(KINT1-1+INT1),INT1=1,NINT1)
        IF(IPRNT.GE.100) THEN
          WRITE(6,*) ' One-electron integrals read in '
          CALL WRTMAT(WORK(KINT1),1,NINT1,1,NINT1)
        END IF
*.2: Two-electron integrals
        IF((INCORE.EQ.1.OR.EXTSPC.EQ.0).AND.ISVMEM.EQ.0) THEN
          READ(LU2INT,'(E22.15)') (WORK(KINT2-1+INT2),INT2=1,NINT2)
        END IF
*.3: Core energy 
        READ(LU2INT,'(E22.15)') ECORE
       ELSE IF (INTIMP .EQ. 5) THEN
*
* ===============
* . SIRIUS format
* ===============
*
        WRITE(6,*) ' Integrals imported from SIRIUS files'
*. Load one-electron integrals
        CALL GETH0S(WORK(KINT1),NTOOB)
*. And two-electron integrals if desired
        IF(INCORE.EQ.1.AND.ISVMEM.EQ.0) THEN
            CALL READMO(WORK(KINT2))
        END IF
       ELSE IF (INTIMP.EQ.8) THEN
*
* =====================
* . Lipkin-Hamiltonian
* =====================
*
        WRITE(6,*) ' setting up Lipkin-Hamiltonian'
        CALL SETINT_LIPK(NACTEL,XLIP_E,XLIP_V,WORK(KINT1),
     &       WORK(KINT2),NINT1,NINT2)
       ELSE IF (INTIMP.EQ.9) THEN
*
* =====================
* . Fusk-integrals
* =====================
*
        WRITE(6,*) ' Integrals set to fusk values '
        CALL SETINT_FUSK(WORK(KINT1),WORK(KINT2),NINT1,NINT2)
       END IF
*       ^ End of switches between different imports of integrals
      END IF
*     ^ End if integrals should be read in
*
      IF(I_DO_REO_ORB.EQ.1) THEN
*. Reorder orbitals 
        CALL REO_INT
      END IF
      ECORE_EXT = ECORE
*
*.Well, one-electron integrals were read in in KINT1, whereas a 
* recent board meeting of the board for LUCIA has decided that
* KH is the right place for the MO 1 electron integrals
      IF(NOINT.EQ.0) THEN 
        CALL COPVEC(WORK(KINT1),WORK(KH),NINT1)
*. and to KINT1O for backwards compatibility
        CALL COPVEC(WORK(KINT1),WORK(KINT1O),NINT1)
       END IF
*
      IF (ISVMEM.EQ.0) THEN
        IF(NOINT.EQ.0) THEN
*. Inactive Fock matrix with contributions from explicitly declared inactive orbitals
            CALL COPVEC(WORK(KINT1),WORK(KHINA),NINT1)
            CALL FISM_OLD(WORK(KHINA),ECC)
            ECORE_INA = ECORE + ECC
            WRITE(6,*) ' ECORE_INA = ', ECORE_INA
*. One-electron integrals with contributions from inactive orbitals and 
*  particle-hole reorganization
            CALL FI(WORK(KINT1),ECORE_HEX,1)
            CALL COPVEC(WORK(KINT1),WORK(KFI),NINT1)
            ECORE_FI = ECORE_HEX
C?          WRITE(6,*) ' First element in FI and H in INTIM=',
C?   &      WORK(KFI), WORK(KINT1O)
        END IF
*. Calculate the FI-alpha and FI-beta matrices
        CALL FI_HS(WORK(KINT1O),WORK(KFI_AL),WORK(KFI_BE),ECORE_AB,1)
*
        ECORE_INI = ECORE
        ECORE_ORIG = ECORE
        ECORE = ECORE + ECORE_HEX
*
        WRITE(6,*) 
     &  ' Core energy: updated and read in ',ECORE, ECORE_ORIG
      END IF
*
C?    WRITE(6,*) ' IDMPIN ', IDMPIN
      IF (IDMPIN.EQ.1 ) THEN
        WRITE(6,*)
     &   ' Integrals written formatted (E22.15) on unit 90'
        REWIND LU90   
*.1: One-electron integrals
        WRITE(LU90,'(E22.15)')
     &   (WORK(KINT1O-1+INT1),INT1=1,NINT1)
*.2: Two-electron integrals
        IF (ISVMEM.EQ.1) THEN
          WRITE(6,*) 'Cannot dump 2el-integral if SAVMEM switch is set!'
          WRITE(6,*) 'Remove that switch and start again...'
          STOP 'INTIM'
        END IF
        WRITE(LU90,'(E22.15)')
     &   (WORK(KINT2-1+INT2),INT2=1,NINT2)
*.3. Core energy 
        WRITE(LU90,'(E22.15)')ECORE_ORIG
*.4  Close to  empty buffer
        CLOSE(LU90)
C       REWIND LU90
*.   Symmetry info etc to LU91
        IF(NOMOFL.EQ.0) KMOAO = KMOAOIN
        CALL DUMP_1EL_INFO(LU91)
      END IF
*
C?    WRITE(6,*) ' INTIM: First integrals in WORK(KINT1) '
C?    LLL = MIN(10,NINT1)
C?    CALL WRTMAT(WORK(KINT1),1,LLL,1,LLL)
C?    WRITE(6,*) ' INTIM: First integrals in WORK(KINT2) '
C?    LLL = MIN(10,NINT2)
C?    CALL WRTMAT(WORK(KINT2),1,LLL,1,LLL)
      
C!    stop ' Jeppe forced my to stop in INTIM '
      RETURN
      END

      SUBROUTINE INTIMM(XINT,MAXSYM)
*
* Import all two electron integrals from MOTRA 2e-file
*
* Jeppe Olsen, Spring of 1992, brewed from M. Fulscher's GETINM routine
*
      IMPLICIT REAL*8(A-H,O-Z)
*.ORBINP
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*.LUNDIO
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
      INTEGER*8 iDAdr
*.Output
      DIMENSION XINT(*)
*. For testing
      Ntest = 0
      write(6,*) ' ************************************************** '
      write(6,*) '    I am dealing with two-electron integrals from   '
      write(6,*) '    MOLCAS. ITRA_ROUTE has been set to 1.           '
      write(6,*) '    I am in INTIMM routine right now!               '
      write(6,*) ' ************************************************** '
      write(6,*) '       >>>>>>    INPUT parameters:    <<<<<         '
      write(6,*) ' File Unit: LuTr2 = ', LuTr2
      write(6,*) ' lTr2Rec          = ', lTr2Rec
      write(6,*) ' iDAdr(mxBatch)   = ',iDAdr
      write(6,*) ' ************************************************** '
      Ioff = 1
      Do 101 Ism = 1, Maxsym
        Do 102 Jsm = 1,Ism
          Do 103 Ksm = 1, ISm
            If(Ism .eq. Ksm ) Then
             LsmMX = Jsm
            Else
             LsmMX = Ksm
            End if
            DO 104 Lsm = 1, LsmMX
            If ( ieor(iSm-1,jSm-1).ne.ieor(kSm-1,lSm-1) )  goto 104
*
              IJbl = Ism*(Ism-1)/2 + Jsm
              KLbl = Ksm*(Ksm-1)/2 + Lsm
              IJKLbl = IJbl*(IJbl-1)/2 + KLbl
*
              nIorb = Ntoobs(Ism)
              nJorb = Ntoobs(Jsm)
              nKorb = Ntoobs(Ksm)
              nLorb = Ntoobs(Lsm)

              If ( Ntest .Ne. 0 ) then
                write(6,*) 
     &          ' ************************************************** '
                Write(6,'(A30,8I4)') 'Ism Jsm Ksm Lsm :',Ism,Jsm,Ksm,Lsm
                write(6,'(A30,4I4)') ' nIorb, nJorb, nKorb, nLorb :', 
     &                       nIorb, nJorb, nKorb, nLorb
              End If
*
              If(Ism.Eq.Jsm) Then
                nIJ = NIorb*(NIorb+1)/2
              Else
                nIJ = NIORB*NJORB
              End If
              If(Ksm.Eq.Lsm) Then
               nKL = nKorb*(nKorb+1)/2
              Else
                nKL = nKorb*nLorb
              End If
*
              If(Ism .Eq. Ksm .And. Jsm .Eq. Lsm ) Then
                nIJKL= nIJ*(nIJ+1)/2
              Else
                nIJKL= nIJ*nKL
              End if
*
              If ( Ntest .Ne. 0 ) then
                write(6,*) 
     &          ' ************************************************** '
                Write(6,*) ' Ism Jsm Ksm Lsm ', Ism,Jsm,Ksm,Lsm
                Write(6,'(A22,2I5)') ' Ioff, nIJKL ', Ioff,nIJKL
              End if
              Ibatch = nbatch(IJKLbl)
              iDisk=iDAdr(iBatch)
              If(Ntest.Ne.0) Write(6,'(A22,3I5)') 
     &        ' IJKLbl iBatch iDisk ',IJKLbl,iBatch,iDisk
*. Loop over records
              nRec = nIJKL/lTr2Rec
              If(Nrec*lTr2Rec.Lt. nIJKL) nRec = nRec + 1
              Ioffo = Ioff
              Do 50 IRec = 1, Nrec
                If ( IRec .Ne. Nrec ) Then
                   Nintrc = lTr2Rec
                Else
                   Nintrc = nIJKL -(Nrec-1)*Ltr2Rec
                End if
C               Call Dafile(LuTr2,2,Xint(Ioff),2*lTr2Rec,Idisk)   
cGLM            Call Dafile(LuTr2,2,Xint(Ioff),2*Nintrc,Idisk)   
                write(6,*) 'Nintrc = ', Nintrc
                Call dDafile(LuTr2,2,Xint(Ioff),Nintrc,Idisk)   
                Ioff = Ioff + Nintrc
   50        Continue
*
             If( Ntest .Ne. 0 ) then
               write(6,*) 'nIJ and nIJKL', nIJ, nIJKL
      write(6,*) ' ************************************************** '
               Write(6,*) ' Integral block '
               If(Ism.Eq.Ksm .And. Jsm. Eq. Lsm ) Then
                 Call Prsym(Xint(Ioffo),nIJ)
               Else
                 Call Wrtmat(Xint(Ioffo),nIJ,nKL,nIJ,nKL)
               End if
             End if
*. Obtain same block with GETINM
C            WRITE(6,*) ' Testing in INTIMM , call to GETINM'
C            CALL GETINM(SCR,2,ISM,2,JSM,2,KSM,2,LSM,0,0,0)
C            WRITE(6,*) ' Matrix obtained from GETINM (form (IK,JL))'
C            CALL WRTMAT(SCR,nIorb*nKorb,nJorb*nLorb,
C    &                   nIorb*nKorb,nJorb*nLorb)
*
  104       Continue
  103     Continue
  102   Continue
  101 Continue
*
      Return
      End
      SUBROUTINE INTPNT(IPNT1,ISL1,IPNT2,ISL2,IPNT2_AB,ISL2_AB)
*
* Pointers to symmetry blocks of integrals
* IPNT1: Pointer to given one-electron block, total symmetric
* ISL1 : Symmetry of last index for given first index, 1 e-
* IPNT2: Pointer to given two-electron block
* ISL1 : Symmetry of last index for given first index, 1 e-
*
*
* In addition pointers to one-electron integrals with general 
* symmetry is generated in WORK(KPGINT1(ISM))
*
* Pointers for similarity transformed Hamiltonian may also be 
* generated
*
* Jeppe Olsen, Update: August 2000
*                       July 2002: _AB pointers added     
c      IMPLICIT REAL*8(A-H,O-Z)
*
* =====
*.Input
* =====
*
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'csmprd.inc'
c*.CSMPRD
c      INTEGER ADASX,ASXAD,ADSXA,SXSXDX,SXDXSX
c      COMMON/CSMPRD/ADASX(MXPOBS,MXPOBS),ASXAD(MXPOBS,2*MXPOBS),
c     &              ADSXA(MXPOBS,2*MXPOBS),
c     &              SXSXDX(2*MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
      INCLUDE 'cintfo.inc'
*
* =======
*. Output
* =======
*
      INTEGER IPNT1(NSMOB),ISL1(NSMOB)
      INTEGER IPNT2(NSMOB,NSMOB,NSMOB),ISL2(NSMOB,NSMOB,NSMOB)
*.0: Pointers to one-integrals, all symmetries, Lower half matrices
      DO ISM = 1, NSMOB
        CALL PNT2DM(1,NSMOB,NSMSX,ADSXA,NTOOBS,NTOOBS,
     &       ISM  ,ISL1,WORK(KPGINT1(ISM)),MXPOBS)
      END DO
*.0.5: Pointers to one-electron integrals, all symmetries, complete form
      DO ISM = 1, NSMOB
        CALL PNT2DM(0,NSMOB,NSMSX,ADSXA,NTOOBS,NTOOBS,
     &       ISM  ,ISL1,WORK(KPGINT1A(ISM)),MXPOBS)
      END DO
*.1: Number of one-electron integrals
      CALL PNT2DM(1,NSMOB,NSMSX,ADSXA,NTOOBS,NTOOBS,
     &            ITSSX,ISL1,IPNT1,MXPOBS)
*.2: two-electron integrals
      CALL PNT4DM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,
     &            ITSDX,ADSXA,SXDXSX,I12S,I34S,I1234S,IPNT2,ISL2,
     &            ADASX,NINT4D)
*
      IF(ISIMTRH.EQ.1) THEN
*. Pointers for similarity transformed Hamiltonian 
        if (isvmem.eq.1) stop 'intpnt!!!!'

        CALL PNT2DM(0,NSMOB,NSMSX,ADSXA,NTOOBS,NTOOBS,
     &         1  ,ISL1,WORK(KPINT1_SIMTRH),MXPOBS)
        I12 = 0
        I34 = 0
        I1234 = 1
        CALL PNT4DM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,
     &              ITSDX,ADSXA,SXDXSX,I12,I34,I1234,
     &              WORK(KPINT2_SIMTRH),ISL2,ADASX,NINT4D)
*. Pointers for open shell 2e A B integrals
        IF(IREFTYP.EQ.2) THEN
          I12 = 0
          I34 = 0
          I1234 = 0
          CALL PNT4DM(NSMOB,NSMSX,MXPOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,
     &         ITSDX,ADSXA,SXDXSX,I12,I34,I1234,
     &         WORK(KPINT2_SIMTRH_AB),ISL2,ADASX,NINT4D)
        END IF
      END IF
C?    write(6,*) ' Memory check INTPNT 2 '
C?    CALL MEMCHK
      RETURN
      END
      FUNCTION IOCTP2(STRING,NEL,ITYP)
*
* Obtain occupation type for STRING .
* For forbidden strings a zero is returned
*
* New version allowing general set of strings
*
      INCLUDE 'mxpdim.inc'
*. Specific input
      INTEGER  STRING(*)
*. General input
      INCLUDE 'strinp.inc'
      INCLUDE 'orbinp.inc'
*
          IF(ITYP.LE.0) THEN
            WRITE(6,*) ' Sorry but you are in trouble '
            WRITE(6,*)
     &      ' String with unallowed number of electrons  in  IOCTP2 '
            WRITE(6,*) ' Number of electrons ', NEL
            STOP ' IOCTP2 error '
          END IF
*. Number of electrons in RAS1 and RAS 3
          IEL1 = 0
          IEL3 = 0
          DO 20 IEL = 1,NEL
            IF(STRING(IEL) .LE. NORB1) IEL1 = IEL1 +1
            IF(NORB1+NORB2+1 .LE. STRING(IEL)) IEL3 = IEL3 + 1
   20     CONTINUE
*. Type
      IF((IEL1.GE.MNRS1(ITYP).AND.IEL1.LE.MXRS1(ITYP)).AND.
     &   (IEL3.GE.MNRS3(ITYP).AND.IEL3.LE.MXRS3(ITYP))) THEN
          ITYP2 = (MXRS1(ITYP)-IEL1)
     &         * (MXRS3(ITYP)-MNRS3(ITYP)+1 )
     &         + IEL3-MNRS3(ITYP)+1
      ELSE
          ITYP2 = 0
      END IF
*
      IOCTP2 = ITYP2
*
      NTEST =  00
      IF ( NTEST .GE.10 ) THEN
        WRITE(6,*) ' From IOCTP2: IEL1 IEL3 ITYP2 ',IEL1,IEL3,ITYP2
      END IF
*
      RETURN
      END
************************************************************************
      Subroutine IPNT2E  
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Initialize the Common /LundIO/                                   *
*
* Only part relvant in connection with use is SIRIUS input that is     *
*     nBatch
*                                                                      *
*     Calling parameters: none                                         *
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
*. Output
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
*----------------------------------------------------------------------*
*     Generate the symmetry block to batch number translation table    *
*----------------------------------------------------------------------*
      iBatch=0
      Do iSym=1,8
        Do jSym=1,iSym
          Do kSym=1,iSym
            mxlSym=kSym
            If ( kSym.eq.iSym ) mxlSym=jSym
            Do lSym=1,mxlSym
              If ( ieor(iSym-1,jSym-1).eq.ieor(kSym-1,lSym-1) ) Then
                ijPair=jSym+iSym*(iSym-1)/2
                klPair=lSym+kSym*(kSym-1)/2
                iSyBlk=klPair+ijPair*(ijPair-1)/2
                iBatch=iBatch+1
                nBatch(iSyBlk)=iBatch
              End If
            End Do
          End Do
        End Do
      End Do
*
      RETURN
      END 
      FUNCTION ISTRNM(IOCC,NORB,NEL,Z,NEWORD,IREORD)
*
* Adress of string IOCC
*
* version of Winter 1990 , Jeppe Olsen
*
      INTEGER Z
      DIMENSION IOCC(*),NEWORD(*),Z(NORB,*)
*
      NTEST = 00
*
      IZ = 1
      DO 100 I = 1,NEL
        IZ = IZ + Z(IOCC(I),I)
  100 CONTINUE
      IF(NTEST.GE.10) WRITE(6,*) ' ISTRNM: IZ = ', IZ
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Z matrix '
        CALL IWRTMA(Z,NORB,NEL,NORB,NEL)
      END IF
*
      IF(IREORD.EQ.0) THEN
        ISTRNM = IZ
      ELSE
        ISTRNM = NEWORD(IZ)
      END IF
*
      IF ( NTEST .GT. 1 ) THEN
        WRITE(6,*) ' STRING'
        CALL IWRTMA(IOCC,1,NEL,1,NEL)
C       WRITE(6,*) ' First two elements of reorder array'
C       CALL IWRTMA(NEWORD,1,2,1,2)
        WRITE(6,*) ' ADRESS OF STRING ',ISTRNM
        WRITE(6,*) ' REV LEX number: ', IZ
      END IF
*
      RETURN
      END
      SUBROUTINE ISWPVE(IVEC1,IVEC2,NDIM)
C
C SWOP INTEGER ARRAYS IVEC1 AND IVEC2
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION IVEC1(1),IVEC2(1)
C
      DO 100 I = 1, NDIM
       IBUF = IVEC1(I)
       IVEC1(I) = IVEC2(I)
       IVEC2(I) = IBUF
  100 CONTINUE
C
      RETURN
      END
      FUNCTION ISYMC1(ICL,IOP,NCL,NOPEN)
*
* Symmmetry of configuration, D2H version
*
      IMPLICIT REAL*8(A-H,O-Z)
*. General input
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*
      INTEGER SYMPRO(8,8)
      DATA  SYMPRO/1,2,3,4,5,6,7,8,
     &             2,1,4,3,6,5,8,7,
     &             3,4,1,2,7,8,5,6,
     &             4,3,2,1,8,7,6,5,
     &             5,6,7,8,1,2,3,4,
     &             6,5,8,7,2,1,4,3,
     &             7,8,5,6,3,4,1,2,
     &             8,7,6,5,4,3,2,1 /
*. Specific input
      INTEGER ICL(*),IOP(*)
*
      ISYM = 1
      DO 100 IEL = 1, NOPEN
        ISYM = SYMPRO(ISYM,ISMFTO(IOP(IEL)))
  100 CONTINUE
*
      ISYMC1 = ISYM
*
      NTEST = 00
      IF(NTEST .NE. 0 ) THEN
        WRITE(6,*) ' ISYMC1, configuration and symmetry '
        CALL IWRTMA(ICL,1,NCL,1,NCL)
        CALL IWRTMA(IOP,1,NOPEN,1,NOPEN)
        WRITE(6,*) ISYM
      END IF
*
      RETURN
      END
      FUNCTION ISYMC2(ICL,IOP,NCL,NOP)
*
* Symmetry of configuration ICONF, D inf h, C inf v, O3 version
*
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*./NONAB/
      LOGICAL INVCNT
      COMMON/NONAB/ INVCNT,NIG,NORASM(MXPOBS),
     &              MNMLOB,MXMLOB,NMLOB,
     &              MXMLST,MNMLST,NMLST,
     &              NMLSX ,MNMLSX,MXMLSX,
     &              MNMLCI,MXMLCI,NMLCI,
     &              MXMLDX,MNMLDX,NMLDX
*
      INTEGER  ICL(*),IOP(*)
*. ML and parity of string
      ML = 0
      IPARI = 1
*. Doubly occupied part
      DO 10 IEL = 1, NCL
        IF(ISMFTO(ICL(IEL)).LE.NMLOB) THEN
          ML = ML + 2*(ISMFTO(ICL(IEL))-1+MNMLOB)
        ELSE
          ML = ML + 2*(ISMFTO(ICL(IEL))-1+MNMLOB-NMLOB)
        END IF
   10 CONTINUE
*. singly occupied part
      DO 20 IEL = 1, NOP
        IF(ISMFTO(IOP(IEL)).LE.NMLOB) THEN
          ML = ML + ISMFTO(IOP(IEL))-1+MNMLOB
        ELSE
          ML = ML + ISMFTO(IOP(IEL))-1+MNMLOB-NMLOB
          IPARI = - IPARI
        END IF
   20 CONTINUE
*
      IF(IPARI.EQ.-1) IPARI = 2
      ISYM  = (IPARI-1) * NMLCI+ ML - MNMLCI + 1
      ISYMC2 = ISYM
*
      NTEST = 00
      IF( NTEST .GE. 1 ) THEN
        WRITE(6,*) ' ISYMC2, configuration and symmetry '
        CALL IWRTMA(ICL,1,NCL,1,NCL)
        CALL IWRTMA(IOP,1,NOP,1,NOP)
        WRITE(6,'(A,3I3)') ' ML, IPARI ISYMC2 ', ML,IPARI,ISYMC2
      END IF
*
      RETURN
      END
      FUNCTION ISYMCN(ICL,IOP,NCL,NOPEN)
*
* Master routine for symmetry of configuration
* with NCL doubly occupied orbitals and NOPEN singly occupied shells
*
*
      IMPLICIT REAL*8(A-H,O-Z)
*. General input ( PNTGRP is used )
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*. Specific input
      INTEGER ICL(*),IOP(*)
      IF(PNTGRP.EQ.1) THEN
*.D2h
        ISYMCN = ISYMC1(ICL,IOP,NCL,NOPEN)
      ELSE IF(PNTGRP.GE.2.AND.PNTGRP.LE.4) THEN
*.Cinfv Dinfh O3
        ISYMCN = ISYMC2(ICL,IOP,NCL,NOPEN)
      ELSE
        WRITE(6,*) ' Sorry PNTGRP option not programmed ', PNTGRP
        WRITE(6,*) ' Enforced stop in ISYMCN '
        STOP 5
      END IF
*
      RETURN
      END
      FUNCTION ISYMS1(STRING,NEL)
*
* Symmmetry of string, D2H version
*
      IMPLICIT REAL*8(A-H,O-Z)
*. General input
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*
      INTEGER SYMPRO(8,8)
      DATA  SYMPRO/1,2,3,4,5,6,7,8,
     &             2,1,4,3,6,5,8,7,
     &             3,4,1,2,7,8,5,6,
     &             4,3,2,1,8,7,6,5,
     &             5,6,7,8,1,2,3,4,
     &             6,5,8,7,2,1,4,3,
     &             7,8,5,6,3,4,1,2,
     &             8,7,6,5,4,3,2,1 /
*. Specific input
      INTEGER STRING(*)
*
      ISYM = 1
      DO 100 IEL = 1, NEL
        ISYM = SYMPRO(ISYM,ISMFTO(STRING(IEL)))
  100 CONTINUE
*
      ISYMS1 = ISYM
*
      NTEST = 00
      IF(NTEST .NE. 0 ) THEN
        WRITE(6,*) ' ISYMS1, String and symmetry '
        CALL IWRTMA(STRING,1,NEL,1,NEL)
        WRITE(6,*) ISYM
      END IF
*
      RETURN
      END
      FUNCTION ISYMS2(STRING,NEL)
*
* Symmetry of string STRING, D inf h, C inf v, O3 version
*
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*./NONAB/
      LOGICAL INVCNT
      COMMON/NONAB/ INVCNT,NIG,NORASM(MXPOBS),
     &              MNMLOB,MXMLOB,NMLOB,
     &              MXMLST,MNMLST,NMLST,
     &              NMLSX ,MNMLSX,MXMLSX,
     &              MNMLCI,MXMLCI,NMLCI,
     &              MXMLDX,MNMLDX,NMLDX
*
      INTEGER   STRING(NEL)
*. ML and parity of string
      MLSTR = 0
      IPARI = 1
      DO 10 IEL = 1, NEL
        IF(ISMFTO(STRING(IEL)).LE.NMLOB) THEN
          MLSTR = MLSTR + ISMFTO(STRING(IEL))-1+MNMLOB
        ELSE
          MLSTR = MLSTR + ISMFTO(STRING(IEL))-1+MNMLOB-NMLOB
          IPARI = - IPARI
        END IF
   10 CONTINUE
*
      IF(IPARI.EQ.-1) IPARI = 2
      ISYM  = (IPARI-1) * NMLST+ MLSTR - MNMLST + 1
      ISYMS2 = ISYM
*
      NTEST = 0
      IF( NTEST .GE. 1 ) THEN
        WRITE(6,*) ' STRING '
        CALL IWRTMA(STRING,1,NEL,1,NEL)
        WRITE(6,'(A,3I3)') ' MLSTR, IPARI ISYMS2 ', MLSTR,IPARI,ISYMS2
      END IF
*
      RETURN
      END
      FUNCTION ISYMST(STRING,NEL)
*
* Master routine for symmetry of string
*
      IMPLICIT REAL*8(A-H,O-Z)
*. General input ( PNTGRP is used )
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*. Specific input
      INTEGER STRING(*)
      IF(PNTGRP.EQ.1) THEN
*.D2h
        ISYMST = ISYMS1(STRING,NEL)
      ELSE IF(PNTGRP.GE.2.AND.PNTGRP.LE.4) THEN
*.Cinfv Dinfh O3
        ISYMST = ISYMS2(STRING,NEL)
      ELSE
        WRITE(6,*) ' Sorry PNTGRP option not programmed ', PNTGRP
        WRITE(6,*) ' Enforced stop in ISYMST '
        STOP 5
      END IF
*
      RETURN
      END
      SUBROUTINE IVCSUM(IA,IB,IC,IFACB,IFACC,NDIM)
*
* Add two (scaled) integer vectors
*
*        IA(*) = IFACB*IB(*) + IFACC*IC(*)
*
      DIMENSION IA(*),IB(*),IC(*)
*
      DO 100 I = 1, NDIM
        IA(I) = IFACB * IB(I) + IFACC * IC(I)
  100 CONTINUE
*
      RETURN
      END
      FUNCTION IWEYLF(NOPEN,MULTS)
C
C NUMBER OF CSF'S WITH NOPEN ORBITALS AND TOTAL MULTIPLICITY
C MULTS ACCORDING TO WEYLS FORMULAE
C
C     (2S+1)/(NOPEN+1) * BION(NOPEN+1/0.5NOPEN-S)
C
      IMPLICIT REAL*8           (A-H,O-Z)
C
      NTEST = 0
 
      IF(NOPEN.EQ.0 .AND. MULTS .EQ. 1 ) THEN
        NCSF = 1
      ELSEIF(MOD(MULTS-1,2) .NE. MOD(NOPEN,2) ) THEN
        NCSF = 0
      ELSEIF(MOD(MULTS-1,2) .EQ. MOD(NOPEN,2) ) THEN
        NCSF = MULTS*IBION(NOPEN+1,(NOPEN+1-MULTS)/2)/(NOPEN+1)
      END IF
C
      IWEYLF = NCSF
C
      IF(NTEST .NE. 0 ) THEN
        WRITE(6,'(A,4I4)')
     &  '  IWEYLF SAYS: NOPEN MULTS NCSF: ', NOPEN,MULTS,NCSF
      END IF
C
      RETURN
      END
 
      SUBROUTINE LFTPOS(CARD,LENGTH)
*
* left position character string CARD
*
* Modified April 2003 !
*
      CHARACTER*102 CARD
*
C     WRITE(6,*) ' INPUT string to LFTPOS '
C     WRITE(6,'(1H ,A)') CARD
*. Number of blanks preceeding keyword
      NBLANK = 0
      DO IPOS = 1, LENGTH
        IF(CARD(IPOS:IPOS).EQ.' ') THEN
          NBLANK = NBLANK + 1
        ELSE 
          GOTO 1001
        END IF
      END DO
 1001 CONTINUE
*. Move string NBLANK characters to the left 
      DO IPOS = NBLANK+1, LENGTH
         IPOSEFF = IPOS-NBLANK
         CARD(IPOSEFF:IPOSEFF) = CARD(IPOS:IPOS)
      END DO
*. Fill end with blanks
      DO IPOS = LENGTH - NBLANK +1, LENGTH
         CARD(IPOS:IPOS) = ' '
      END DO
C
C     DO 100 IPOS = 1, LENGTH
C      IF(CARD(IPOS:IPOS).NE.' ') THEN
C        IEFF = IEFF + 1
C        IF(IEFF.NE.IPOS) CARD(IEFF:IEFF) = CARD(IPOS:IPOS)
C      END IF
C 100 CONTINUE
C.Fill end with trailing blanks
C     DO 200 IPOS = IEFF+1,LENGTH
C       CARD(IPOS:IPOS) = ' '
C 200 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Left adjusted character string '
        WRITE(6,'(1H ,A)') CARD
      END IF
*
      RETURN
      END
      
      PROGRAM LUCIA
*
* L U C I A 
*
*
* CI for program for :FCI
*                     RASCI
*                     MRSDCI
*                     GASCC
*                     GAS GAS GAS GAS GAS GAS
*
* Written by Jeppe Olsen , winter of 1991
*                          GAS version in action summer of 95
*                          CC added fall of 99 / winter of 00
*
*
c      IMPLICIT REAL*8(A-H,O-Z)
*. Parameters for dimensioning
c      INCLUDE 'mxpdim.inc'
*.Memory
      INCLUDE 'wrkspc.inc'
      LOGICAL CONV_F
*.File numbers
      INCLUDE 'clunit.inc'
*.Print flags
      INCLUDE 'cprnt.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'crun.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'chktyp.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'cfinal_e.inc'
      INCLUDE 'cshift.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'fragmol.inc'
      INCLUDE 'cecore.inc'
      INCLUDE 'mv7task.inc'
      INCLUDE 'vb.inc'
*. Added dec. 2011
      INCLUDE 'spinfo.inc'
      COMMON/CENOT/E0
      LOGICAL LTARGET
*.Scratch: A character line 
      CHARACTER*72 CARD
      CHARACTER*3  CCFORM_REM
*. A temp block for locating the SETVEC use/misuse
      COMMON/XXTEST/ISETVECOPS(10)
*
      NTEST = 0
*
      CALL ISETVC(ISETVECOPS,0,10)
* 1 => MATML7
* 2 => RSBB2BN2 
* 3 => RSBB2A 
*.
C     CALL TEST_DEC15 
      CALL QENTER('REST ')
*. Initialize counters for kernel routines 
      CALL KERNEL_ROU_STAT_INI
*.    No floating point underflow
      CALL XUFLOW
*. Assign diskunits
      CALL DISKUN
*. Header
      CALL WRTHD(LUOUT)
* ======================================================
* 1: .Read input,insert defaults and cross check input
* ======================================================
      CALL INPCTL(LUIN,LUOUT)
*. Static memory, initialize
      KBASE = 1
      KADD = MAXMEM
      CALL MEMMAN(KBASE,KADD,'INI   ',IDUMMY,'DUMMY ')
*. Ratio beteeen real and integer word length
      CALL  ZIRAT
*. From shells to orbitals
      CALL ORBINF(LUOUT,IPRORB)
*. We now have the number of active orbitals. Set up the 
*  default map for MINMAX space = identity map
      IF(I_DO_NORTCI.EQ.0) THEN
        CALL ISTVC2(IREO_MNMX_OB_NO,0,1,NACOB)
        CALL ISTVC2(IREO_MNMX_OB_ON,0,1,NACOB)
      END IF
*. Symmetry of reference (pt not active)
      IF(PNTGRP.GT.1) CALL MLSM(IREFSM,IREFPA,IREFSM,'CI',1)
*. Number of string types
      CALL STRTYP_GAS(IPRSTR)
*
*. Allocate some inital memory - pointers to construct pointers
*
      CALL ALLO_ALLO_0
*. Divide orbital spaces into inactive/active/secondary
      CALL GASSPC
*. Symmetry information
      CALL SYMINF(IPRORB)
*. Number of integrals
      CALL INTDIM(IPRORB)
*. Set up dimensions of general integral lists
      I_DO_INT_G = 1
      IF(I_DO_INT_G.EQ.1) THEN
        CALL Z_TYP_EI_LISTS
        CALL INTDIM_G(IPRORB)
      END IF
* 
* =================================================================================
* Memory for prototype information for CSFs and dimensions of the various CI spaces
* =================================================================================
*
*. I will play around with subpspace configurations
      I_DO_SBCNF = 1
*
* ===================================================
*. Generate occupation classes of the compound space
* ===================================================
*
      CALL GEN_OCCLS_FOR_CISPAC(IGSOCC(1,1), IGSOCC(1,2), -1,
     &     int_mb(KIOCCLS),NOCCLS_MAX)
*. Info in the sub occupation classes and configurations
      IF(I_DO_SBCNF.EQ.1) THEN
        CALL INFO_OCSBCLS
      END IF

      IF(NOCSF.EQ.0) THEN
*
       CALL PROTO_CSF_DIM
*
* ========================================================
*. Generate the dimensions of the occupation sub classes 
* ========================================================
*
*. And the occupation subclasses (given number of elecs in given space)
       CALL GEN_DIM_SBCNF(dbl_mb(KNSBCNF),dbl_mb(KIBSBCNF),
     &      dbl_mb(KLSBCNF),dbl_mb(KOGOCSBCLS),dbl_mb(KMNOPOCSBCL))
*. Number of CSF's SD, and Confs per configuration type using subconf info
       IF(I_DO_SBCNF.EQ.1) THEN
         DO ICISPC = 1, NCISPC
C                DIM_CISPACE_FROM_SBCNF(ICISPC,NCNFOPSM)
            CALL DIM_CISPACE_FROM_SBCNF(ICISPC,
     &           NCONF_PER_OPEN_GN(1,1,ICISPC))
         END DO
       END IF ! I_DO_SBCNF
*. 
*
*. Information about the Valence bond expansions, defined as 
*. min-max spaces
*
       IF(I_DO_NORTCI.NE.0) THEN
       
* inactive orbitals in Gaspaces
        N_ORB_CONF =  NACOB 
        N_EL_CONF = NACTEL
        IB_ORB_CONF = NINOB + 1
        WRITE(6,*)
        WRITE(6,*) ' Active orbitals in the order they are occupied: '
        CALL IWRTMA(IREO_MNMX_OB_NO,1,N_ORB_CONF,1,N_ORB_CONF)
        WRITE(6,*)
        CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'MN_MNX')
        CALL MEMMAN(KLMINOCC,NACOB,'ADDL  ',1,'MINOCC')
        CALL MEMMAN(KLMAXOCC,NACOB,'ADDL  ',1,'MAXOCC')
        DO IVBSP = 0, NVBGNSP
          IF(IVBSP.EQ.0) THEN
            CALL GET_DIM_MINMAX_SPACE(VB_REFSPC_MIN,VB_REFSPC_MAX,
     &           IREO_MNMX_OB_NO,N_ORB_CONF,IREFSM,
     &           NCONF_L,NCSF_L,NSD_L,NCM_L,
     &           LCONFOCC_L,NCONF_AS_L)
          ELSE
            CALL GET_DIM_MINMAX_SPACE(
     &           VB_GNSPC_MIN(1,IVBSP),VB_GNSPC_MAX(1,IVBSP),
     &           IREO_MNMX_OB_NO,N_ORB_CONF,IREFSM,
     &           NCONF_L,NCSF_L,NSD_L,NCM_L,
     &           LCONFOCC_L,NCONF_AS_L)
          END IF
         NVB_CONF(IVBSP) = NCONF_L
         NVB_CSF(IVBSP)  = NCSF_L
         NVB_CM(IVBSP)   = NCM_L
        END DO
        CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'MN_MNX')
*
        IF(NTEST.GE.0) THEN
         WRITE(6,*) 
     &   ' Info on dimensions of VB MINMAX spaces (0=reference space)'
         WRITE(6,*)
         WRITE(6,*) '   Space   NCONF   NCSCF      NCM '
         WRITE(6,*) ' ================================='
         DO ISP = 0, NVBGNSP
           WRITE(6,'(3X,I4,3(1X,I8))') 
     &     ISP, NVB_CONF(ISP), NVB_CSF(ISP), NVB_CM(ISP) 
         END DO
        END IF! NTest is large enough
       END IF! There are General VB spaces
*
*.Info on number of CSF's, SD, conf per occupation type
*. This may give problems for non-orthogonal CI with 
* inactive orbitals in Gaspaces
       N_ORB_CONF =  NACOB 
       N_EL_CONF = NACTEL
       IB_ORB_CONF = NINOB + 1
*
       IF(I_DO_SBCNF.EQ.0) THEN
*. Two local arrays for max/min for occupation for orbs
        IDUM = 0
        CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'MN_MNX')
        CALL MEMMAN(KLMINOCC,NACOB,'ADDL  ',1,'MINOCC')
        CALL MEMMAN(KLMAXOCC,NACOB,'ADDL  ',1,'MAXOCC')
*
        IF(ICNFBAT.EQ.1) THEN
          NCONF_MAX = 0
          NCSF_MAX = 0
          NSD_MAX = 0
          NCM_MAX = 0
          NCONF_AS_MAX = 0
          LCONFOCC_MAX = 0
          DO ICISPC = 1, NCISPC
*. Max/min for orbitals
            CALL MXMNOC_GAS(WORK(KLMINOCC),WORK(KLMAXOCC),NGAS,NOBPT,
     &                     IGSOCCX(1,1,ICISPC),IGSOCCX(1,2,ICISPC),
     &                     IPRCSF)
*. Number of parameters for this Max/min space
            CALL GET_DIM_MINMAX_SPACE(
     &           WORK(KLMINOCC),WORK(KLMAXOCC),
     &           NACOB,IREFSM, NCONF_L, NCSF_L, NSD_L,NCM_L,LCONFOCC_L,
     &           NCONF_AS_L)
            IF(NTEST.GE.10) THEN
              WRITE(6,'(A)') 
     &        ' ICISPC, NCONF_L, NCSF_L, NSD_L, NCM_L,NCONF_AS_L= ' 
              WRITE(6,'(1H ,I3,5(2X,I9))') 
     &          ICISPC, NCONF_L, NCSF_L, NSD_L, NCM_L,NCONF_AS_L
            END IF
*
            NCONF_MAX = MAX(NCONF_MAX,NCONF_L)
            NCSF_MAX  = MAX(NCSF_MAX,NCSF_L)
            NSD_MAX = MAX(NSD_MAX,NSD_L)
            NCM_MAX = MAX(NCM_MAX,NCM_L)
            NCONF_AS_MAX = MAX(NCONF_AS_MAX,NCONF_AS_L)
            LCONFOCC_MAX = MAX(LCONFOCC_MAX,LCONFOCC_L)
*
            NCONF_PER_SYM_GN(IREFSM, ICISPC) = NCONF_L
            NCSF_PER_SYM_GN(IREFSM, ICISPC) = NCSF_L
            NSD_PER_SYM_GN(IREFSM, ICISPC) = NSD_L
            NCM_PER_SYM_GN(IREFSM, ICISPC) = NCM_L
            NCONF_ALL_SYM_GN(ICISPC) = NCONF_AS_L
          END DO
          IF(NTEST.GE.10) WRITE(6,*) ' LCONFOCC_MAX = ', LCONFOCC_MAX
       END IF ! ICNFBAT = 1
       CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'MN_MNX')
      END IF ! I_DO_SBCNF = 0
      END IF! NOCSF = 0
*
*. Set up the basespace for the various occupation classes
C          Z_BASSPC_FOR_ALL_OCCLS(IOCCLS,NOCCLS,IBASSPC)
      CALL Z_BASSPC_FOR_ALL_OCCLS(int_mb(KIOCCLS),NOCCLS_MAX,
     &      WORK(KBASSPC))
*. Dimensions for all occupation classes

*
* =========================================
* Construct the information on the strings 
* =========================================
*
      CALL STRINF_GAS(WORK,IPRSTR)
*
*. Dimensions for all occupation classes
      IF(NOCSF.EQ.0) THEN 
C?      CALL GEN_INFO_FOR_ALL_OCSBLS
        CALL GEN_INFO_FOR_ALL_OCCLS(I_DO_SBCNF)
      END IF
*
* ========================================
*. Dimension of the various CI-expansions
* ========================================
*
      CALL LCISPC(IPRCIX)
*
* ============================================
*. Dimension of explicit Hamiltonian subspace
* ============================================
*
*
      IF(ISBSPC_SEL.EQ.0) THEN
       MXP1_MAX = 0
      ELSE IF(ISBSPC_SEL.EQ.3) THEN
*. Subspace is specified as a given CI-space, determine max dim
*. (all symmetries allowed)
        IF(NOCSF.EQ.1) THEN
          MXP1_MAX = 0
          DO ISM = 1, NSMOB
            NCM = XISPSM(ISM,ISBSPC_SPC)
C                 XISPSM(1,ICI)
            MXP1_MAX = MAX(MXP1_MAX,NCM)
          END DO
        ELSE
          MXP1_MAX = IMNMX(NCSF_PER_SYM_GN(1,ISBSPC_SPC),NSMST)
        END  IF
      ELSE IF(ISBSPC_SEL.EQ.4) THEN
*. Subspace defined by a minmax
C GET_NSD_MINMAX_SPACE(MIN_OCC,MAX_OCC,ISYM,MS2X,MULTSX, NSD,NCM,NCSF,NCONF)
         IF(NOCSF.EQ.1) THEN
           MULTS = MS2 + 1
         END IF
         CALL GET_NSD_MINMAX_SPACE(
     &        ISBSPC_MINMAX(1,1),ISBSPC_MINMAX(1,2),ISBSPC_ORB,IREFSM,
     &        MS2,MULTS,NSD,NCM,NCSF,NCONF,LOCC)
         IF(NOCSF.EQ.1) THEN
          MXP1_MAX = NCM
          NCONF_SUB = 0
          LOCC_SUB = 0
         ELSE
          MXP1_MAX = NCSF 
          NCONF_SUB = NCONF
          LOCC_SUB = LOCC
         END IF
         MXP1 = MXP1_MAX 
         NP1 = MXP1_MAX
      END IF
      WRITE(6,*) ' MXP1_MAX, fresh = ', MXP1_MAX

*
* =========================
* Allocate static memory 
* =========================
*
      CALL ALLO_ALLO
*
* =========================================================================
* Construct information about prototype configurations, CSFs, CMBs and SDs
* =========================================================================
*
      IF(NOCSF.EQ.0) THEN 
*. The prototype information
        CALL CSDTMT_GAS(int_mb(KDFTP),int_mb(KCFTP),dbl_mb(KDTOC),IPRCSF)
      END IF
      IF(NOCSF.EQ.0.AND.I_DO_SBCNF.EQ.1) THEN
*. Construct the occupations of the various subconfigurations
       CALL GEN_OCC_SBCNF(dbl_mb(KNSBCNF),dbl_mb(KIBSBCNF),
     &      dbl_mb(KOGOCSBCLS),dbl_mb(KMNOPOCSBCL),dbl_mb(KKOCSBCNF))
C           GEN_OCC_SBCNF(NSBCNF_FOR_OP_SM,IBSBCNF_FOR_OP_SM,
C    &                         IOGOCSBCLS,MINOPFSPCLS,KOCSBCNF)

      END IF
*. Cont
*
* =====================================
* Generate the initial sets of orbitals
* =====================================
*
* Procedure is:
*   1) Obtain MO-AO expansion of MOs defining the initial set of orbitals
*   2) Obtain MO-AO expansion in which calculations will be performed (or start)
*
* Part 1: Read in from environment
*
      LMOMO = NDIM_1EL_MAT(1,NTOOBS,NTOOBS,NSMOB,0)
      IF(NOMOFL.EQ.0) THEN
        CALL GET_CMOAO_ENV(WORK(KMOAOIN))
        CALL COPVEC(WORK(KMOAOIN),WORK(KMOAO_ACT),LMOMO)
*. And as long as nothing else has happened, it is also
*. the output MO's
        CALL COPVEC(WORK(KMOAOIN),WORK(KMOAOUT),LMOMO)
      END IF
* Part 2
* The set of starting MOs: MO-AO expansion and MOMO expansion.
*.Again, the starting MOs are those in which integrals are imported,
*.whereas initial MOs are used in the calculations.

*. Transformation between starting MO's and initial orbitals
*. Obtain AO integrals SAO- will be used later
*
      IF(INI_MO_TP.EQ.3) THEN
*. Well, the starting MOs are the initial MOs, so the corresponding transformation matrix 
*  is just the unit matrix 
        ONE = 1.0D0
        CALL SETDIA_BLM(WORK(KMOMO),ONE,NSMOB,NTOOBS,0)
      ELSE
*. Extended form for starting orbitals:
*. Expansion of starting MOs in AOs in WORK(KMOAOUT)
*  Expansion of starting MOs in initial MOs in MOMO
        XDUM = 2810.1979
        CALL GET_HSAO(XDUM,WORK(KSAO),0,1)
        CALL GET_CMOINI_GEN(WORK(KMOAOUT),WORK(KMOMO),WORK(KMOAOIN))
*. Let the starting orbitals be the orbitals in action
        CALL COPVEC(WORK(KMOAOUT),WORK(KMOAOIN),LMOMO)
        CALL COPVEC(WORK(KMOAOUT),WORK(KMOAO_ACT),LMOMO)
*. Test Orthonormality of starting orbitals
*.a) Overlap matrix in lower half form
        CALL MEMMAN(KLSMO,LMOMO,'ADDL  ',2,'SMO_L  ')
        CALL GET_SMO(WORK(KMOAOIN),WORK(KLSMO),0)
*.b) Check deviation from unit matrix
C       BLK_CHECK_UNI_MAT(UNI,NBLK,LBLK,XMAX_DIFF_DIAG,XMAX_DIFF_OFFD)
        CALL BLK_CHECK_UNI_MAT(WORK(KLSMO),NSMOB,NTOOBS,
     &       XMAX_DIFF_DIAG,XMAX_DIFF_OFFD)
        WRITE(6,*) ' Deviation of SMO from unit matrix: '
        WRITE(6,'(A,6X,E8.3)') 
     &  ' Largest difference of off-diagonal from zero  ',
     &   XMAX_DIFF_OFFD
        WRITE(6,'(A,11X,E8.3)') 
     &  ' Largest difference of diagonal from one  ', 
     &   XMAX_DIFF_DIAG
      END IF
*
* ======================================================================
* Integrals over Starting orbitals: stored in both KINT_2EMO, KINT_2EINI 
* ======================================================================
*
      IF(NOINT.EQ.0) THEN
*. 2-electron integrals are saved in KINT_2EMO
        CALL INTIM(IPRORB)
*. copy to list of initial 2 e integrals
        CALL COPVEC(WORK(KINT_2EMO),WORK(KINT_2EINI),NINT2)
      ELSE
        WRITE(6,*) ' No integrals imported '
      END IF
*. Print MO-coefs (after INTIM for getting labels)
      IF(NOMOFL.EQ.0.AND.IPRORB.GE.1) THEN
        WRITE(6,*) ' MO-AO transformation matrix of starting orbitals'
        CALL PRINT_CMOAO(WORK(KMOAOIN))
      END  IF
*
* ============================================================================
* Super symmetry of orbitals and perhaps supersymmetry-reordering of orbitals
* ============================================================================
*
*
      IF(I_USE_SUPSYM.EQ.1) THEN
*
*. a: General supersymmetry information for BASIS
*
*. Obtain number of super-symmetry irreps for basis functions
*. (N_SUPSYM_IRREP,MAXL stored in ORBINP)
        CALL GET_MAX_SUPSYM_IRREP
*. Obtain info on the supersymmetries of the basis functions
        CALL GET_SUPSYM_INFO
      ELSE
* Set GENSMOB arrays to the standard symmetry
        NGENSMOB = NSMOB
        CALL ICOPVE(NTOOBS,NBAS_GENSMOB,NSMOB)
        CALL ICOPVE(ITOOBS,IBBAS_GENSMOB,NSMOB)
C            ISTVC2(IVEC,IBASE,IFACT,NDIM)
        CALL ISTVC2(ISTA_TO_GENSM_REO,0,1,NTOOB)
        CALL ISTVC2(IACT_TO_GENSM_REO,0,1,NTOOB)
        CALL ISTVC2(ISTASM_FOR_GENSM,0,1,NGENSMOB)
        DO IGAS = 0, NGAS+1
         DO ISYM = 1, NGENSMOB
          NGAS_GNSYM(ISYM,IGAS) = NOBPTS_GN(IGAS,ISYM)
         END DO
        END DO
        CALL ICOPVE(ISMFSO,WORK(KMO_GNSYM),NTOOB)
        CALL REO_ACT_ORB_TO_GNSM(
     &     WORK(KMO_GNSYM),WORK(KIREO_GNSYM_TO_TS_ACOB))
      END IF! Supersymmetry
*
      IF(I_USE_SUPSYM.EQ.1) THEN
       IF(I_DO_HF.EQ.0) THEN
*
*. save super symmetry order for standard
        CALL ICOPVE(WORK(KISUPSYM_FOR_BAS),WORK(KMO_STA_SUPSYM),NTOOB)
*. b. Info on supersymmetry of actual orbitals 
        CALL SUPSYM_FROM_CMOAO(WORK(KMOAOIN),WORK(KISUPSYM_FOR_BAS),
     &                         WORK(KMO_SUPSYM))
C           SUPSYM_FROM_CMOAO(CMOAO,ISUPSYM_FOR_BAS,ISUPSYM_FOR_MOS)
        CALL ICOPVE(WORK(KMO_SUPSYM),WORK(KMO_GNSYM),NTOOB)
        CALL ICOPVE(WORK(KMO_SUPSYM),WORK(KMO_ACT_SUPSYM),NTOOB)
        IF(IPRORB.GE.5) THEN
          WRITE(6,*) ' Actual supersymmetry of initial orbitals '
          CALL IWRTMA3(WORK(KMO_SUPSYM),1,NTOOB,1,NTOOB)
        END IF
       ELSE
* Hartree-Fock: The MO's will start out with the supersymmetry of 
* the basis functions
        CALL ICOPVE(WORK(KISUPSYM_FOR_BAS),WORK(KMO_SUPSYM),NTOOB)
       END IF ! Hartree-Fock will not be called
*
* c. Info on REQUIRED supersymmetry order of the orbitals and various
*    reorder arrays
*
       CALL ORDER_GAS_SUPSYM_ORBITALS
       CALL ICOPVE(WORK(KMO_SUPSYM),WORK(KMO_GNSYM),NTOOB)
*. Supersymmetry in occupation order
        CALL GET_OCC_ORDER_SUPSYM(WORK(KMO_OCC_SUPSYM))
*. And save in general symmetry arrays
       DO IGAS = 0, NGAS + 1
        CALL ICOPVE(NGAS_SUPSYM(1,IGAS),NGAS_GNSYM(1,IGAS),N_SUPSYM)
       END DO
*. Mapping of active orbitals from general symmetry to type order
       CALL REO_ACT_ORB_TO_GNSM(
     &      WORK(KMO_GNSYM),WORK(KIREO_GNSYM_TO_TS_ACOB))
*. Reordering array from actual to supersymmetry-blocked order.
C       GET_IACT_TO_GENSM_REO(IACT_TO_GENSM_REO,
C    &  ISTA_TO_GENSM_REO, MO_STA_TO_ACT_REO, NTOOB)
        CALL GET_IACT_TO_GENSM_REO(IACT_TO_GENSM_REO,
     &  ISTA_TO_GENSM_REO, WORK(KMO_STA_TO_ACT_REO),NTOOB)
*
       IF(I_DO_HF.EQ.0) THEN
*
*. We have read in some orbital and will start out with these. However, check first
*. the orbitals have the specified symmetry ordering
*
        WRITE(6,*) ' CMO_ORD == ', CMO_ORD
*
        IF(CMO_ORD.EQ.'STA') THEN
*
* The orbitals were assumed to be in standard form, check this
*
         WRITE(6,*) ' It is assumed that input orbitals fulfill: '
         WRITE(6,*) 
     &   '    Input molecular orbital have super-symmetry '
         WRITE(6,*) 
     &   '    Input molecular orbitals are in standard super-sym order'
C                IS_I1_EQ_I2(I1,I2,NDIM)
         IDENT = IS_I1_EQ_I2(WORK(KISUPSYM_FOR_BAS),
     &                       WORK(KMO_SUPSYM),NTOOB)
         IF(IDENT.EQ.0) THEN
           WRITE(6,*) ' Error: Input orbitals are not in expected order'
           WRITE(6,*) ' Required standard supersymmetry-order '
           CALL IWRTMA3(WORK(KISUPSYM_FOR_BAS),1,NTOOB,1,NTOOB)
           STOP ' Error: Input orbitals are not in expected order'
         END IF
         WRITE(6,*) ' Input orbitals tested and were in correct order'
        ELSE IF (CMO_ORD.EQ.'OCC') THEN
         WRITE(6,*) ' It is assumed that input orbitals fulfill: '
         WRITE(6,*) 
     &   '    Input molecular orbital have super-symmetry '
         WRITE(6,*) 
     &   '    Input molecular orbitals are in GAS supersym order'
*. Reorder the orbitals to standard order and check that this order is correct
*. Reform 
         CALL REFORM_CMO(WORK(KMOAOIN),2,WORK(KMOAOUT),1)
*. Determine symmetry
         CALL SUPSYM_FROM_CMOAO(WORK(KMOAOUT),WORK(KISUPSYM_FOR_BAS),
     &                         WORK(KMO_SUPSYM))
*. Compare
         IDENT = IS_I1_EQ_I2(WORK(KISUPSYM_FOR_BAS),
     &                       WORK(KMO_SUPSYM),NTOOB)
         IF(IDENT.EQ.0) THEN
           WRITE(6,*) ' Error: Input orbitals are not in expected order'
           WRITE(6,*) ' Obtained symmetry of reordered orbitals '
           CALL IWRTMA3(WORK(KMO_SUPSYM),1,NTOOB,1,NTOOB)
           WRITE(6,*) ' Required order of basis functions '
           CALL IWRTMA3(WORK(KISUPSYM_FOR_BAS),1,NTOOB,1,NTOOB)
           STOP ' Error: Input orbitals are not in expected order'
         END IF
*. Restore
         CALL ICOPVE(WORK(KMO_GNSYM),WORK(KMO_SUPSYM),NTOOB)
*
        ELSE 
*
* CMO_ORD was not specified as STA or OCC, obtain order and reform to occ
*
*. Obtain array going from input orbitals to expected output order
C             REO_2SUPSYM_ORDERS(ISUPSYM1,ISUPSYM2,IREO12)
         CALL REO_2SUPSYM_ORDERS(WORK(KMO_OCC_SUPSYM),
     &        WORK(KMO_ACT_SUPSYM),WORK(KIREO_INI_OCC))
*. Reform MOMO and MOAO to OCC order
C        CALL REO_CMOAO(WORK(KMOMO),WORK(KMOAO_ACT),
C    &        WORK(KIREO_INI_OCC),1,1)
C        CALL REO_CMOAO(WORK(KMOAOIN),WORK(KMOAO_ACT),
C    &        WORK(KIREO_INI_OCC),1,1)
         CALL REO_CMOAO(WORK(KMOMO),WORK(KMOAO_ACT),
     &        WORK(KIREO_INI_OCC),1,2)
         CALL REO_CMOAO(WORK(KMOAOIN),WORK(KMOAO_ACT),
     &        WORK(KIREO_INI_OCC),1,2)
         LEN_C =  LEN_BLMAT(NSMOB,NTOOBS,NTOOBS,0)
         CALL COPVEC(WORK(KMOAO_ACT),WORK(KMOAOUT),LEN_C)
*. Update the MO-MO transformation matrix
CERR     CALL BLK_SET_REORDER_XMAT(WORK(KMOMO),NSMOB,NTOOBS,
CERR &        WORK(KIREO_INI_OCC))
*. Check that orbitals now are in occ order
*. Determine symmetry
         WRITE(6,*) ' Supersymmetry of reordered MOs: '
         CALL SUPSYM_FROM_CMOAO(WORK(KMOAOUT),WORK(KISUPSYM_FOR_BAS),
     &                         WORK(KMO_ACT_SUPSYM))
*. Compare
         IDENT = IS_I1_EQ_I2(WORK(KMO_OCC_SUPSYM),
     &                       WORK(KMO_ACT_SUPSYM),NTOOB)
         IF(IDENT.EQ.0) THEN
           WRITE(6,*) ' Error: Reordered orbitals are not in occ order'
           WRITE(6,*) ' Obtained symmetry of reordered orbitals '
           CALL IWRTMA3(WORK(KMO_ACT_SUPSYM),1,NTOOB,1,NTOOB)
           WRITE(6,*) ' Required order '
           CALL IWRTMA3(WORK(KMO_OCC_SUPSYM),1,NTOOB,1,NTOOB)
           STOP ' Error: Reordered orbitals are not in expected order'
         END IF
         CMO_ORD = 'OCC'
        END IF 
*
        IF(CMO_ORD.EQ.'STA') THEN
*
*. Reorder to OCC if standard form
*
         CALL REO_CMOAO(WORK(KMOAOIN),WORK(KMOAO_ACT),
     &        WORK(KMO_STA_TO_ACT_REO),1,1)
         LEN_C =  LEN_BLMAT(NSMOB,NTOOBS,NTOOBS,0)
         CALL COPVEC(WORK(KMOAO_ACT),WORK(KMOAOIN),LEN_C)
*. Update the MO-MO transformation matrix
         CALL BLK_SET_REORDER_XMAT(WORK(KMOMO),NSMOB,NTOOBS,
     &        WORK(KMO_STA_TO_ACT_REO))
         CMO_ORD = 'OCC'
        END IF
*
        IF(IPRORB.GE.2) THEN
           WRITE(6,*) 
     &     ' Input orbitals reordered to requested super-symmetry order'
           CALL PRINT_CMOAO(WORK(KMOAOIN))
         END IF
*
         IF(IPRORB.GE.5) THEN
           WRITE(6,*) ' MO-MO transformation matrix '
           CALL APRBLM2(WORK(KMOMO),NTOOBS,NTOOBS,NSMOB,0)
         END IF 
       END IF !I_DO_HF = 0
*
       IF(IPRORB.GE.5) THEN
         WRITE(6,*) ' Input orbitals in shell form '
         CALL PRINT_CMO_AS_SHELLS(WORK(KMOAOIN),2)
C             PRINT_CMO_AS_SHELLS(CMO,IFORM)
       END IF
*. Analyze and align eqivalent components of deg irreps
C            ANA_SUBSHELLS_CMO(CMO,IFORM,XMAX,MAXIRR,MAXSHL,
C    &                              IALIGN)
        CALL ANA_SUBSHELLS_CMO(WORK(KMOAOIN),2,XMAXDF,MAXIRRDF,
     &       MAXSHLDF,1)
        WRITE(6,*) ' Test analyze of aligned orbitals '
        CALL ANA_SUBSHELLS_CMO(WORK(KMOAOIN),2,XMAXDF,MAXIRRDF,
     &       MAXSHLDF,0)

*
      END IF! Supersymmetry
*
* =====================================================================================================
* Allocate space,set pointers for the active lists of two-electron integrals and copy full list of ints
* =====================================================================================================
*
      IF(ITRA_ROUTE.EQ.2) THEN
        CALL Z_ACT_INTLISTS
* Integrals over initial orbitals in arrays defined by new order (third list of integrals...)
*
        IF(IE2LIST_FULL.EQ.0) THEN
         WRITE(6,*) 
     &   ' Address for full integral list has not been defined..'
         STOP       
     &   ' Address for full integral list has not been defined..'
        END IF
*
        IF(NTEST.GE.1000)
     &  WRITE(6,*) ' TEST: IE2LIST_FULL = ', IE2LIST_FULL
        IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_FULL))
        IE2ARRAY_A = IE2ARR_F
        IF(NTEST.GE.1000)
     &  WRITE(6,*) ' TEST: IE2ARR_F = ', IE2ARR_F
        NINT2_F = NINT2_G(IE2ARR_F)
        KINT2_F = KINT2_A(IE2ARR_F)
        CALL COPVEC(WORK(KINT_2EINI),WORK(KINT2_F),NINT2_F)
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) 
     &    ' Full integral list copied to WORK(KINT2(IE2ARR_F))'
          WRITE(6,*) ' TEST: KINT2_F, NINT2_F = ', KINT2_F, NINT2_F
          WRITE(6,*) ' WORK(KINT_2EINI),WORK(KINT2_F)',
     &                 WORK(KINT_2EINI),WORK(KINT2_F)
        END IF
* Tell program to work with full two-electron integral list (until further notice)
        IE2LIST_A = IE2LIST_FULL
        IOCOBTP_A = 1
        INTSM_A = 1
        CALL PREPARE_2EI_LIST
        I12S_A = 1
        I34S_A = 1
        I1234S_A = 1
        CALL FLAG_ACT_INTLIST(IE2LIST_FULL)
      END IF ! End if new integral transformation and storage is in place
*
* ======================================================
*. Perform integral transformation to starting orbitals
* ======================================================
*
*. Pt not done when doing nonorthogonal CI (done later..)
*
C     IF(I_DO_NORTCI.EQ.0.AND.INI_MO_TP.NE.3) THEN
      IF(I_DO_NORTCI.EQ.0) THEN
        WRITE(6,*) 
     &  ' Integrals will be transformed to new initial orbitals'
*. Flag type of integral list to be obtained: Pt complete list of integrals
        IE2LIST_A = IE2LIST_FULL
        IOCOBTP_A = 1
        INTSM_A = 1
        KKCMO_I = KMOMO
        KKCMO_J = KMOMO
        KKCMO_K = KMOMO
        KKCMO_L = KMOMO
        IH1FORM = 1
        IH2FORM = 1
*. Integrals will be fetched from KINT_2EMO
        KINT2 = KINT_2EMO
        CALL TRAINT
        WRITE(6,*) ' Integral transformation completed '
*. And overwrite two-electron integrals
*. Move 2e- integrals to KINT_2EMO 
        IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_FULL))
        NINT2_F = NINT2_G(IE2ARR_F)
        KINT2_F = KINT2_A(IE2ARR_F)
        CALL COPVEC(WORK(KINT2_F),WORK(KINT_2EMO),NINT2_F)
        IF(NTEST.GE.10000) THEN
          WRITE(6,*) ' NINT2_F = ', NINT2_F
          WRITE(6,*) ' Integrals transformed to KINT_2EMO'
          CALL WRTMAT(WORK(KINT_2EMO),1,NINT2_F,1,NINT2_F)
        END IF
*. one-electron integrals to KINT1O
        CALL COPVEC(WORK(KINT1),WORK(KINT1O),NINT1)
*. And to KH
        CALL COPVEC(WORK(KINT1),WORK(KH),NINT1)
        IF(IPRORB.GE.100) THEN
          WRITE(6,*) ' One-electron integrals after transf'
          CALL APRBLM2(WORK(KH),NTOOBS,NTOOBS,NSMOB,1)
        END IF
*. The integrals corresponds now to the new initial orbitals, reset MOMO
*. matrix to one
        ONE = 1.0D0
        CALL SETDIA_BLM(WORK(KMOMO),ONE,NSMOB,NTOOBS,0)
*
* ================================
*. Construct inactive Fock matrix
* ================================
*
        IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_FULL))
        KINT2_FSAVE = KINT2_A(IE2ARR_F)
        KINT2_A(IE2ARR_F) = KINT_2EMO
C            FI_FROM_INIINT(FI,CINI,H,EINAC,IHOLETP)
        CALL FI_FROM_INIINT(WORK(KFI),WORK(KMOMO),WORK(KH),
     &                      ECORE_HEX,3)
        CALL COPVEC(WORK(KFI),WORK(KINT1),NINT1)
        ECORE = ECORE_ORIG + ECORE_HEX
        IF(NTEST.GE.10000) THEN
          WRITE(6,*) ' ECORE_ORIG, ECORE_HEX, ECORE(2) ',
     &                 ECORE_ORIG, ECORE_HEX, ECORE
        END IF
*. Clean up
        KINT2 = KINT_2EMO
        KINT2_A(IE2ARR_F) = KINT2_FSAVE
      END IF
*
      IF(NOINT.EQ.1.OR.ENVIRO(1:4).EQ.'NONE') THEN
        WRITE(6,*) ' End of calculation without integrals'
        CALL QSTAT 
        STOP' End of calculation without integrals'
      END IF
*
*. Product expansion wave-function: completely different world 
      IF( I_DO_PRODEXP.EQ.1) THEN
        CALL LUCIA_PRODEXP
        STOP ' Enforced stop after LUCIA_PRODEXP '
      END IF
*
* ======================================================================================
* We are now finished with the initialization and start to prepare for the calculations
* ======================================================================================
*
*. Last space where CI vectors were stored
*
      ISTOSPC = 0
      IF(IRESTR.EQ.1) ISTOSPC = 1
      IRESTR_ORIG=IRESTR
*
* =======================================================
*. Restarted Lambda calculations need special attention
* =======================================================
*
*  Restart is realized in in SECOND calculation 
*  First calculation is used to establish H0
*  we do therefore first copy vectors on LUC to LUSC39
*  so we can restart from this file later 
*
      IF(IRESTR.EQ.1.AND.XLAMBDA.NE.1.0D0) THEN
        CALL MEMMAN(IDUM,IDUM,'MARK ',IDUM,'VCSAVE')
        WRITE(6,*) ' Restart vectors from previous run'
        WRITE(6,*) ' will be saved on LUSC39   '
*
        IF(ICISTR.EQ.1) THEN
          LBLK = XISPSM(IREFSM,1)
        ELSE
          LBLK = -1
        END IF
*
        IF(ICISTR.EQ.1) THEN
          LBLOCK = XISPSM(IREFSM,1)
        ELSE IF (ICISTR.EQ.2) THEN
          LBLOCK = MXSB
        ELSE IF (ICISTR.EQ.3) THEN
          LBLOCK = MXSOOB
        END IF
        CALL MEMMAN(KVEC1,LBLOCK,'ADDS  ',2,'VEC1  ')
*
        CALL REWINO(LUC)
        CALL REWINO(LUSC39)
        DO JROOT = 1, NROOT
          CALL COPVCD(LUC,LUSC39,WORK(KVEC1),0,LBLK)
        END DO
        CALL MEMMAN(IDUM,IDUM,'FLUSM',IDUM,'VCSAVE')
      END IF
*     ^ End of special handling of restarted calc with lambda modified op.
*
* ==============================================================
* ==============================================================
* Loop over GAS spaces and perform calcalations: CI, PERT, ....
* ==============================================================
* ==============================================================
*
      ISKIPEI_INI = ISKIPEI
      I_HAVE_DONE_CC = 0
      I_HAVE_DONE_GAS = 0
      IIUSEH0P = 0
      ICHKTP = 1
      EREF_CI = 3006.0D0
      I_AM_DOING_NORT = 0
      INI_NORT = 1
      IVBSPC_PREV  = -1
*
*. Default task of MV7 is to SIGMA calculations
      CMV7TASK = 'SIGMA ' 
      DO JCMBSPC = 1, NCMBSPC
        ISKIPEI = 0
        IF(JCMBSPC.EQ.1) THEN
          MPORENP_E = 0
        ELSE
          MPORENP_E = MPORENP
        END IF
*
        IF(I_DO_NORTCI.EQ.1.AND.JCMBSPC.NE.1) THEN
*. At the moment we cannot expand CI calculations with minmax definitions, so
          IRESTR = 0
          WRITE(6,*) 
     &    ' IRESTR set to zero for first calculation in comb. space'
        END IF
*
        WRITE(6,*) 
        WRITE(6,*) 
        WRITE(6,*) 
        WRITE(6,'(15X,A)') 
     &  '********************************'
        WRITE(6,'(15X,A)') 
     &  ' ******************************'
        WRITE(6,*) 
        WRITE(6,'(15X,A,I3)') 
     &  '   Calculations in space ', JCMBSPC
        WRITE(6,*) 
        WRITE(6,'(15X,A)') 
     &  ' ******************************'
        WRITE(6,'(15X,A)') 
     &  '********************************'
        WRITE(6,*) 
        WRITE(6,*) 
        WRITE(6,*) 
        WRITE(6,*) 
C       WRITE(6,'(A,I3)')
C    &  ' Number of calculation in this CI space ', NSEQCI(JCMBSPC)
*. Special treatment of lambda calc in first calc
        IF(IRESTR.EQ.1.AND.XLAMBDA.NE.1.0D0.AND.JCMBSPC.EQ.1) THEN
          WRITE(6,*) ' Remember No restart in calc 1 (Lambda calc)'
          IRESTR = 0
        END IF
*     
        I_EXPAND = 1
        IF(XLAMBDA.NE.1.0D0 .AND.JCMBSPC.GT.1) THEN      
          WRITE(6,*) ' =================================='
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) '   Modified operator will be used'
          WRITE(6,*) ' =================================='
CMO       IF(JCMBSPC.EQ.2) THEN
*. Obtain modified operator for lambda calculations
CMO         WRITE(6,*) ' Operator will be modified '
CMO         CALL GENH1(XLAMBDA)
CMO         CALL SCLH2(XLAMBDA)
CMO       END IF
        END IF
*
        IF(JCMBSPC.EQ.2.AND.IRESTR_ORIG.EQ.1.AND.XLAMBDA.NE.1.0D0)  THEN
*.Obtain restart vectors for Lambda calculations from LUSC39
          WRITE(6,*) ' Restart vectors will be copied to LUC'
          WRITE(6,*) ' CI will restart with vectors from prev. calc'
          IF(ICISTR.EQ.1) THEN
            LBLK = XISPSM(IREFSM,2)
          ELSE
            LBLK = -1
          END IF
          CALL MEMMAN(IDUM,IDUM,'MARK ',IDUM,'VCSAVE')
          IF(ICISTR.EQ.1) THEN
            LBLOCK = XISPSM(IREFSM,2)
          ELSE IF (ICISTR.EQ.2) THEN
            LBLOCK = MXSB
          ELSE IF (ICISTR.EQ.3) THEN
            LBLOCK = MXSOOB
          END IF
          CALL MEMMAN(KVEC1,LBLOCK,'ADDS  ',2,'VEC1  ')
*. Copy vectors from LUSC39 to LUC
          CALL REWINO(LUC)
          CALL REWINO(LUSC39)
          DO JROOT = 1, NROOT
            CALL COPVCD(LUSC39,LUC,WORK(KVEC1),0,LBLK)
          END DO
          CALL MEMMAN(IDUM,IDUM,'FLUSM',IDUM,'VCSAVE')
*. No expansion should follow
          I_EXPAND = 0
*. But normal restart
          IRESTR = 1
        END IF
*       ^ End of section copying restart vectors from LUSC39
*. Loop over Calculations in given space
        DO JSEQ = 1,  NSEQCI(JCMBSPC)
        CARD = CSEQCI(JSEQ,JCMBSPC)
*
*. Signal whether this is a target model
        LTARGET = .FALSE.
        IF (JCMBSPC.EQ.ITGSPC.AND.JSEQ.EQ.ITGCLC) THEN
          LTARGET = .TRUE.
          WRITE(6,*) 'This calculation was declared as TARGET'
          WRITE(6,*) 'Information for subsequent programs will'//
     &               ' be dumped at the end of this run!'
        END IF
*
*
* =======================
*. Expansion of CI VECTORS
* =======================
*
*. Note: For internal contracted calculations, expansion to 
*. currrent space has been removed 
* Note: for nonorthogonal CI calculations, expansion is currently inactive
          ISTOSPCP = ISTOSPC
          IF((CARD(1:2).EQ.'CI'.OR.CARD(1:4).EQ.'PERT'
     &        .OR.CARD(1:2).EQ.'CC'.OR.CARD(1:2).EQ.'ICXX'.OR.
     &         CARD(1:6).EQ.'GEN_CC'.OR.
     &         CARD(1:3).EQ.'TCC'.OR.CARD(1:3).EQ.'ECC'.OR.
     &         CARD(1:3).EQ.'UCC'.OR.CARD(1:3).EQ.'VCC'
     &         .OR.CARD(1:5).EQ.'MCSCF')
     &   .AND. JCMBSPC.NE.1.AND.ISTOSPC.NE.JCMBSPC.AND.I_EXPAND.EQ.1 
     &   .AND. I_HAVE_DONE_GAS.EQ.1)THEN
*
*. Restart from previous spaces ( Assuming a progressing sequence:
*  spaces are just added, not subtracted )
*  ( Used for perturbation, CI, CC, MCSCF  calculations)
*
            LUIN = LUC
            LUOUT = LUSC1
            IF(ICISTR.EQ.1) THEN
              LBLK = XISPSM(IREFSM,JCMBSPC)
            ELSE
              LBLK = -1
            END IF
C           WRITE(6,*) ' LBLK = ', LBLK
C           WRITE(6,*) ' Vectors will be expanded '
            I_SKIP_CIV = 0
            IF(I_SKIP_CIV.EQ.0) THEN
             IF(NOCSF.EQ.1) THEN
              CALL EXPCIV(IREFSM,ISTOSPC  ,LUIN,JCMBSPC,LUOUT,
     &                    LBLK,LUSC2,
     &                    NROOT,1,IDC,IPRDIA) 
             ELSE
              CALL EXPCIV_CSF(IREFSM,ISTOSPC,LUIN,JCMBSPC,LUOUT,
     &                    LBLK,NROOT,1,IDC,IPRDIA) 
             END IF! NOCSF switch
            ELSE 
              WRITE(6,*) ' Expansion of CI vectors skipped ! '
              WRITE(6,*) ' Expansion of CI vectors skipped ! '
              WRITE(6,*) ' Expansion of CI vectors skipped ! '
              WRITE(6,*) ' Expansion of CI vectors skipped ! '
              WRITE(6,*) ' Expansion of CI vectors skipped ! '
            END IF
*. Last space where vectors were stored
            ISTOSPC = JCMBSPC
            ISKIPEI = ISKIPEI_INI
*. Expanded vector will be used as initial vector in the
*. zero space calculation. Tell next CI to restart from
*. CI vectors
            IRESTR = 1
          END IF
*         ^ End of Expansion section 
            IF(IUSEH0P.EQ.1.AND.JCMBSPC.NE.1) THEN 
*. Expand file containing zero order vectors 
            WRITE(6,*) ' Zero-order vector will be expanded '
            CALL EXPCIV(IREFSM,ISTOSPCP,LUSC51,JCMBSPC,LUSC52,
     &                  -1,LUSC53,
     &                  1,1,IDC,IPRDIA)
            END IF
* ==
* CI        
* ==
          IF(CARD(1:2).EQ.'CI') THEN
            IF(JSEQ.EQ.1.AND.JCMBSPC.EQ.2.AND.IRST2.EQ.0) THEN
*. No restart from previous vectors - IRST2 has been set to zero
              IRESTR = 0
              WRITE(6,*) ' No restart from previous vectors'
            END IF
*. Good old normal CI !!!!
*. do CI in space JCMBPSC
            MAXIT_SAVE = MAXIT
            MAXIT = ISEQCI(JSEQ,JCMBSPC)
            IROOT_SEL_SAVE = IROOT_SEL
            IF(IRESTR.EQ.0) THEN
*. Root selection is not used in  iterative procedure
              IROOT_SEL = 0
*. Initial CI, should a larger number of roots be used
              NROOT_SAVE = NROOT
              MXCIV_SAVE = MXCIV
              NROOT = INI_NROOT
              MXCIV = MAX(3*NROOT,MXCIV_SAVE)
              NCNV_RT_SAVE = NCNV_RT
              NCNV_RT = INI_NROOT
              WRITE(6,*) ' INI_*ROOT option in action '
            END IF ! special setting for initial CI
*
            CALL GASCI(IREFSM,JCMBSPC,IPRDIA,IIUSEH0P,
     &                 MPORENP_E,
     &                 EREF,ERROR_NORM_FINAL,CONV_F)
*. If special settings has used modify and reset
            IROOT_SEL = IROOT_SEL_SAVE 
            IF(IRESTR.EQ.0) THEN
*. Reset parameters
              NROOT = NROOT_SAVE
              MXCIV = MXCIV_SAVE
              NCNV_RT = NCNV_RT_SAVE
            END IF
*
            I_HAVE_DONE_GAS = 1
*. A CI calculation has been performed, the default is 
* now to restart following calcs from this
            IRESTR = 1
            EREF_CI = EREF
*
            E_FINAL_T(JSEQ,JCMBSPC) = EREF
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = ERROR_NORM_FINAL
            CONV_T(JSEQ,JCMBSPC) = CONV_F
*
            MAXIT = MAXIT_SAVE
*. Transform CI coefficients to CC form
            IF(I_DO_CI_TO_CC.EQ.1.AND.JCMBSPC.GT.1) THEN
              CALL CI_TO_CC_REFRM(LU_CC_FROM_CI,LUC,JCMBSPC,IREFSM)
            END IF
*. Modified one-electron operator in first it
            IF(XLAMBDA.NE.1.0D0 .AND.JCMBSPC.EQ.1) THEN      
*. 
              IF(IUSEH0P.EQ.1) THEN
*. Perturbation operator will be of type PFP + E0|0><0|,
*. obtain E0 and save |0>
              CALL GET_E0(E0,EREF)
              WRITE(6,*) ' zero-order energy =', E0
*. Prepare for perturbation calculation 
              SHIFT = 0.0D0
              IPROJ = 1
              CALL MEMMAN(IDUM,IDUM,'MARK ',IDUM,'VCSAVE')
              LBLOCK = MXSOOB
              CALL MEMMAN(KVEC1,LBLOCK,'ADDS  ',2,'VEC1  ')
*. Copy vectors from LUC to LUSC51
              CALL REWINO(LUC)
              CALL REWINO(LUSC51)
              LBLK = -1
              CALL SKPVCD(LUC,IH0ROOT-1,WORK(KVEC1),1,LBLK) 
              CALL COPVCD(LUC,LUSC51,WORK(KVEC1),0,LBLK)
C?            WRITE(6,*) ' First vector copied to LUSC51'
C?            CALL WRTVCD(WORK(KVEC1),LUSC51,1,LBLK)
              CALL MEMMAN(IDUM,IDUM,'FLUSM',IDUM,'VCSAVE')
              IIUSEH0P = 1
             END IF
             IF(IUSEH0P.EQ.0.AND.MPORENP.EQ.1) THEN
*. Obtain modified operator for lambda calculations
                WRITE(6,*) ' Operator will be modified '
                CALL GENH1(XLAMBDA)
                CALL SCLH2(XLAMBDA)
              END IF
            END IF
*. Transform CI coeffficients
            IF(ITRACI.NE.0) THEN
              WRITE(6,*) ' Control will be transferred to TRACI_CTL'
              CALL TRACI_CTL
            END IF
*. Last space where vectors were stored
            ISTOSPC = JCMBSPC
*. Test orbital Hessian routines
C            CALL TEST_E12
*
* =======================
* Vector free Calculation
* =======================
*
          ELSE IF (CARD(1:7).EQ. 'VECFREE'     ) THEN
         
            WRITE(6,'(A,I3)') ' Vector free calculation at level ',
     &                          -ISEQCI(JSEQ,JCMBSPC)
            LEVEL =  -ISEQCI(JSEQ,JCMBSPC)
*. Should the first order correction be explicitly constructed ?
            IF(IC1DSC.LE.0) THEN
              LU1EFF = 0
            ELSE
              LU1EFF = LUHC
            END IF
            WRITE(6,*) ' LU1EFF in MAIN ', LU1EFF
*
            CALL DIRDIR(JCMBSPC-1,JCMBSPC,IREFSM,LUC,LEVEL,EREF,
     &                  LUSC2,LUSC3,LU1EFF,EOUT)
            E_FINAL_T(JSEQ,JCMBSPC) = EOUT
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = 0.0D0
            CONV_T(JSEQ,JCMBSPC) = .TRUE.
*
* ================================
* General perturbation calculation
* ================================
*
          ELSE IF (CARD(1:5).EQ.'PERTU'      ) THEN 
            WRITE(6,'(A)') ' Perturbation calculation '
            CALL PERTCTL(IREFSM,JCMBSPC,EREF,EFINAL)
            E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = 0.0D0
            CONV_T(JSEQ,JCMBSPC) = .TRUE.

            IF( NPROP.GT.0) THEN
*. Perturbation expansion of properties 
              CALL PROP_PERT(LUC,LUSC36,NPERT,IREFSM,JCMBSPC)
C                  PROP_PERT(LU0,LUN,N,ISM,ISPC)
            END IF
            IF(IPTFOCK.NE.0) THEN
              CALL PTFOCK(LUC,LUSC36,NPERT,IREFSM,JCMBSPC)
            END IF
*. also perturbation expansion of EKT ??
            IF(IPTEKT.EQ.1) THEN
              WRITE(6,*) ' Perturbation expansion of EKT '
              WRITE(6,*) ' ============================= '
              CALL PTEKT(LUC,LUSC36,NPERT,IREFSM,JCMBSPC)
            END IF
          ELSE IF(CARD(1:2).EQ.'CC' .OR. 
     &            CARD(1:6).EQ.'GEN_CC'.OR.
     &            CARD(1:3).EQ.'TCC'.OR.
     &            CARD(1:3).EQ.'ECC'.OR.
     &            CARD(1:3).EQ.'VCC'.OR.
     &            CARD(1:3).EQ.'UCC'    ) THEN
*
* ============================
* Coupled Cluster calculation 
* ============================
*
            MAXIT_SAVE = MAXIT
            MAXIT = ISEQCI(JSEQ,JCMBSPC)
*
            IF(I_HAVE_DONE_CC.EQ.1.OR.IRES_EXC.EQ.1) THEN
             II_RES_EXC   = 1
            ELSE
             II_RES_EXC   = 0
            END IF
*
            IF(I_HAVE_DONE_CC.EQ.1.OR.I_RESTRT_CC.EQ.1) THEN
             II_RESTRT_CC = 1
            ELSE
             II_RESTRT_CC = 0
            END IF
*. Transfer expanded cc wf to LUC in the last CC calc
            IF(I_DO_CC_TO_CI.EQ.1.AND.
     &         JCMBSPC.EQ.LAST_CC_SPC.AND.JSEQ.EQ.LAST_CC_RUN) THEN
              I_TRANS_WF = 1
            ELSE
              I_TRANS_WF = 0
            END IF
*. If CMPCCI has been specified, comparison between CC vector expanded 
*  in the actual CI space and a CI vector on LU17 is 
*  done in the last space where 
            IF(JCMBSPC.EQ.LAST_CC_SPC.AND.JSEQ.EQ.LAST_CC_RUN.AND.
     &        I_DO_CMPCCI.EQ.1) THEN
              I_TRANS_WF = 1
              II_DO_CMPCCI = 1
            ELSE
              II_DO_CMPCCI = 0
            END IF
       
            WRITE(6,*) ' LAST_CC_SPC, LAST_CC_RUN = ',
     &                   LAST_CC_SPC, LAST_CC_RUN 
            WRITE(6,*) ' JCMBSPC, JSEQ, I_TRANS_WF = ', 
     &                   JCMBSPC, JSEQ, I_TRANS_WF
            WRITE(6,*) ' I_DO_CMPCCI, II_DO_CMPCCI = ',
     &                   I_DO_CMPCCI, II_DO_CMPCCI
*
            IF(I_HAVE_DONE_CC.EQ.0.AND.IRFROOT.NE.1) THEN
*. In initial CI, the reference root was an excited root, 
*. obtain this state as first root on LUC
              CALL MEMMAN(IDUM,IDUM,'MARK ',IDUM,'VCSAVE')
              LBLOCK = MXSOOB
              CALL MEMMAN(KVEC1,LBLOCK,'ADDS  ',2,'VEC1  ')
*. Copy vectors from LUC to LUSC51
              CALL REWINO(LUC)
              CALL REWINO(LUSC51)
              LBLK = -1
              CALL SKPVCD(LUC,IRFROOT-1,WORK(KVEC1),1,LBLK) 
              CALL COPVCD(LUC,LUSC51,WORK(KVEC1),0,LBLK)
              CALL REWINO(LUC)
              CALL REWINO(LUSC51)
              CALL COPVCD(LUSC51,LUC,WORK(KVEC1),0,LBLK)
              CALL MEMMAN(IDUM,IDUM,'FLUSM',IDUM,'VCSAVE')
            END IF
*
COLD        IF(CARD(1:2).EQ.'CC') THEN
COLD          CALL LUCIA_CC(IREFSM,JCMBSPC,IPRDIA,EREF,II_RESTRT_CC,
COLD &                      I_TRANS_WF)
            IF( CARD(1:2) .EQ. 'CC' .OR. CARD(1:6).EQ.'GEN_CC'.OR.
     &          CARD(1:3).EQ.'TCC'.OR.
     &          CARD(1:3).EQ.'ECC'.OR.
     &          CARD(1:3).EQ.'VCC'.OR.
     &          CARD(1:3).EQ.'UCC'    ) THEN
*. From Febr 2003, The use of CC implies that the newer codes are used, 
*   where the T space equals the CISPACE so 
              IF(CARD(1:2) .EQ. 'CC') THEN
                ITSPC = JCMBSPC
              ELSE 
                ITSPC  = ISEQCI2(JSEQ,JCMBSPC)
              END IF
              CCFORM_REM(1:3) = CCFORM(1:3)
              CCFORM(4:6) = '   '
              I_DO_NEWCCV_REM = I_DO_NEWCCV
              ISIMTRH_REM     = ISIMTRH
              IUSE_PH_REM     = IUSE_PH
              IF (CARD(1:4).EQ.'UCC2'.OR.
     &            (CARD(1:3).EQ.'ECC'.AND.
     &             CARD(4:4).NE.' '.AND.
     &             CARD(4:4).NE.'-')) THEN
                CCFORM(1:4) = CARD(1:4)
                ! optimized orbitals?
                IF (CARD(5:6).EQ.'-O') THEN
                  I_OOCC = 1
                ELSE
                  I_OOCC = 0
                END IF
                IF (CARD(5:6).EQ.'-H') THEN
                  I_OBCC = 1  ! triggering OO/Brueckner-hybrid
                ELSE
                  I_OBCC = 0
                END IF
                ! brueckner orbitals?
                IF (CARD(5:6).EQ.'-B') THEN
                  I_BCC = 1
                ELSE
                  I_BCC = 0
                END IF
              ELSE IF (CARD(1:3).EQ.'TCC'.OR.
     &            CARD(1:3).EQ.'ECC'.OR.
     &            CARD(1:3).EQ.'VCC'.OR.
     &            CARD(1:3).EQ.'UCC'    ) THEN
                CCFORM(1:3) = CARD(1:3)
                ! optimized orbitals?
                IF (CARD(4:5).EQ.'-O') THEN
                  I_OOCC = 1
                ELSE
                  I_OOCC = 0
                END IF
                IF (CARD(4:5).EQ.'-H') THEN
                  I_OBCC = 1  ! triggering OO/Brueckner-hybrid
                  IF (CARD(1:3).EQ.'VCC') THEN
                    WRITE(6,*)
     &                   'VCC-H makes no sense, changing to VCC-O!'
                    I_OBCC=0
                    I_OOCC=1
                  END IF
                ELSE
                  I_OBCC = 0
                END IF
                ! brueckner orbitals?
                IF (CARD(4:5).EQ.'-B') THEN
                  I_BCC = 1
                ELSE
                  I_BCC = 0
                END IF
              END IF
              IF (
     &            CARD(1:3).EQ.'VCC'.OR.
     &            CARD(1:3).EQ.'UCC'    ) THEN
                CCFORM(1:3) = CARD(1:3)
                I_DO_NEWCCV = 0
                ISIMTRH = 0
                IUSE_PH = 0
              END IF

              CALL LUCIA_GENCC(IREFSM,ITSPC,JCMBSPC,IPRDIA,
     &                         II_RESTRT_CC,I_TRANS_WF,II_RES_EXC,
     &                         II_DO_CMPCCI,LTARGET,
     &                         E_FINAL,ERROR_NORM_FINAL,CONV_F)
              I_HAVE_DONE_GAS = 1
*
              E_FINAL_T(JSEQ,JCMBSPC) = E_FINAL
              ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = ERROR_NORM_FINAL
              CONV_T(JSEQ,JCMBSPC) = CONV_F
*
              MAXIT = MAXIT_SAVE 
              CCFORM(1:3) = CCFORM_REM(1:3)
              I_DO_NEWCCV = I_DO_NEWCCV_REM
              ISIMTRH     = ISIMTRH_REM
              IUSE_PH     = IUSE_PH_REM
            END IF
            I_HAVE_DONE_CC = 1
*
* ==========================================
* Internal contracted CI, PT or CC calculation 
* ==========================================
*
          ELSE IF(CARD(1:2).EQ.'IC') THEN
*. reference energy is energy of previous reference state CI-calculation-
            EREF = EREF_CI
            IF(IEI_VERSION .EQ. 0) THEN 
              CALL LUCIA_IC(ISTOSPC,JCMBSPC,CARD,EREF,0,
     &                      EFINAL,CONV_F,VNFINAL)
            ELSE
              CALL LUCIA_IC_EI(ISTOSPC,JCMBSPC,CARD,EREF,0,
     &                         EFINAL,CONV_F,VNFINAL)
            END IF
            E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
            CONV_T(JSEQ,JCMBSPC) = CONV_F
          ELSE IF(CARD(1:3).EQ.'GIC') THEN
*. Generalized internal contraction....
*. Assumed no EI first time around
            CALL LUCIA_GIC(CARD,EREF,EFINAL,CONV_F,VNFINAL)
            E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
            CONV_T(JSEQ,JCMBSPC) = CONV_F
          ELSE IF(CARD(1:6).EQ.'CUMULA') THEN
*
* ==========================================
* Calculation of cumulant matrices 
* ==========================================
*
            CALL LUCIA_IC(JCMBSPC,JCMBSPC,CARD,EREF,1,
     &                    EFINAL,CONVER,VNFINAL)
   
          ELSE IF(CARD(1:7).EQ.'TWOBODY') THEN
c            CALL LUCIA_LSQUARE(ISEQCI2(JSEQ,JCMBSPC),
c     &                JCMBSPC,ISEQCI(JSEQ,JCMBSPC))
            CALL LUCIA_GTBCE(ISEQCI2(JSEQ,JCMBSPC),
     &                JCMBSPC,ISEQCI(JSEQ,JCMBSPC))
          ELSE IF(CARD(1:6).EQ.'SP_MCL' ) THEN
             WRITE(6,*) ' Next stop: SP_MCLR '
             CALL SP_MCLR(IREFSM,JCMBSPC)
          ELSE IF(CARD(1:5).EQ.'MCSCF'  ) THEN
*
* ===================
*. MCSCF calculation
* ===================
* 
             MAXMAC = ISEQCI(JSEQ,JCMBSPC)
             MAXMIC = ISEQCI2(JSEQ,JCMBSPC)
             CALL LUCIA_MCSCF(JCMBSPC,MAXMAC,MAXMIC,
     &                        EFINAL,CONV_F,VNFINAL)
*. Check subshells for supersymmetry
             IF(I_USE_SUPSYM.EQ.1) THEN
                IF(IPRORB.GE.5) THEN
                  WRITE(6,*) ' Input orbitals in shell form '
                  CALL PRINT_CMO_AS_SHELLS(WORK(KMOAO_ACT),2)
                END IF
*. Analyze eqivalence of components
                 CALL ANA_SUBSHELLS_CMO(WORK(KMOAO_ACT),2,XMAXDF,
     &                MAXIRRDF,MAXSHLDF,0)
             END IF
*
             I_HAVE_DONE_GAS = 1
             E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
             ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
             CONV_T(JSEQ,JCMBSPC) = CONV_F
             ISTOSPC = JCMBSPC
          ELSE IF(CARD(1:6).EQ.'NORTCI' ) THEN
*
* ==========================================
* Nonorthogonal CI calculations 
* ==========================================
*
            MAXIT = ISEQCI(JSEQ,JCMBSPC)
            IVBGNSP = ISEQCI2(JSEQ,JCMBSPC)
            I_DO_NONORT_MCSCF = 0
*. Forced cold start for test
COLD        IRESTR = 0
            CALL LUCIA_NORT(I_DO_NONORT_MCSCF,
     &           JCMBSPC,EFINAL,CONV_F,VNFINAL,INI_NORT,IVBGNSP,
     &           IVBGNSP_PREV)
            INI_NORT = 0
            E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
            CONV_T(JSEQ,JCMBSPC) = CONV_F
*. Allow restart for CI in given combination space
             IRESTR = 1
             IVBGNSP_PREV = IVBGNSP
          ELSE IF(CARD(1:6).EQ.'NORTMC' ) THEN
*
* ==========================================
* Nonorthogonal MCSCF calculations 
* ==========================================
*
            MAXIT_MAC = ISEQCI(JSEQ,JCMBSPC)
            MAXIT_MIC = ISEQCI2(JSEQ,JCMBSPC)
C           MAXIT = MAXIT_MIC
            IVBGNSP = 0
            I_DO_NONORT_MCSCF = 1
            CALL LUCIA_NORT(I_DO_NONORT_MCSCF,
     &           JCMBSPC,EFINAL,CONV_F,VNFINAL,INI_NORT,IVBGNSP)
            INI_NORT = 0
            E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
            CONV_T(JSEQ,JCMBSPC) = CONV_F
*. Allow restart for CI in given combination space
             IRESTR = 1
             IVBGNSP_PREV = IVBGNSP

          ELSE IF (CARD(1:2).EQ.'HF') THEN
*
* =========================
* Hartree-Fock calculation 
* =========================
*
            MAXIT_HF = ISEQCI(JSEQ,JCMBSPC)
            WRITE(6,*) ' MAXIT_HF = ', MAXIT_HF
            CALL LUCIA_HF(IREFSM,JCMBSPC,MAXIT_HF,
     &            E_HF,E1_FINAL,CONV_F)
            E_FINAL_T(JSEQ,JCMBSPC) = E_HF
            CONV_T(JSEQ,JCMBSPC) = CONV_F
            ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = -1.0D0
          ELSE IF(CARD(1:6).EQ.'AKBKCI') THEN
*
* ================
* AKBKCI calculation 
* ================
*
            write(6,*) 'I am entering AKBK'
            CALL AKBKCI(JCMBSPC,IPRDIA,
     &                 EREF,ERROR_NORM_FINAL,CONV_F)
             E_FINAL_T(JSEQ,JCMBSPC) = EREF
             ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = 
     &       ERROR_NORM_FINAL
             CONV_T(JSEQ,JCMBSPC) = CONV_F
          ELSE IF(CARD(1:6).EQ.'AKBKMC') THEN
*
* ======================
* AKBKMCSCF calculation 
* ======================
*
             MAXMAC = ISEQCI(JSEQ,JCMBSPC)
             MAXMIC = ISEQCI2(JSEQ,JCMBSPC)
             CALL LUCIA_AKMCSCF(IREFSM,JCMBSPC,MAXMAC,MAXMIC,
     &                        EFINAL,CONV_F,VNFINAL)
             I_HAVE_DONE_GAS = 1
             E_FINAL_T(JSEQ,JCMBSPC) = EFINAL
             ERROR_NORM_FINAL_T(JSEQ,JCMBSPC) = VNFINAL
             CONV_T(JSEQ,JCMBSPC) = CONV_F
             ISTOSPC = JCMBSPC
   
          END IF
*         ^ End of switch between types of calculations
        END DO
*       ^ End of loop over calculations in a given expansion
*
      END DO
*     ^ End of loop over CI Expansions
*
*  ====================================
*. Transition properties of final state
*  ====================================
*
      IF(ITRAPRP.NE.0) THEN
        WRITE(6,*) ' Transition properties of final states '
        WRITE(6,*) ' and states on file LUEXC ( unit 17 ) ' 
        WRITE(6,*) ' will now be calculated '
        WRITE(6,*)
        CALL TRAPRP
      END IF
      IF(IGENTRD.EQ.1) THEN
        WRITE(6,*) ' General transition matrix between final state '
        WRITE(6,*) ' and state on file LUEXC ( unit 17 ) ' 
        WRITE(6,*) ' will now be calculated '
        WRITE(6,*)
        IVGSIGDEN = 2
        CALL VGSIGDEN_M(IVGSIGDEN)
      END IF
*
* =================
* Final set of MO's
* =================
*
      IF(NOMOFL.EQ.0.AND.IFINMO.NE.0) THEN
*. Create final set of mo's
        WRITE(6,*) ' I am going to call MOROT '
        KMOAO = KMOAOUT
        INEW = 1
        IF(INEW.EQ.0) THEN
          CALL MOROT(IFINMO)
        ELSE
          CALL MOROT_GS(IFINMO)
        END IF
      END IF
      IF(I_USE_SUPSYM.EQ.1.AND.I_DO_GAS.EQ.1) THEN
*. We have orbitals in the supersymmetry order specified by NGAS_SP
*. If the output orbitals should be used for another supersymmetry 
*. calculation, the orbitals should be rearranged to standard order.
*, The form is specied by I_NEGLECT_SUPSYM_FINAL_MO
        IIFORM = 2
        IF(I_NEGLECT_SUPSYM_FINAL_MO.EQ.0) THEN
*. Reorder back to standard order
C              REO_CMOAO(CIN,COUT,IREO,ICOPY,IWAY)
          CALL REO_CMOAO(WORK(KMOAOIN),WORK(KMOAOUT),
     &         WORK(KMO_STA_TO_ACT_REO),0,2)
          CALL REO_CMOAO(WORK(KMOMO),WORK(KMOAO_ACT),
     &         WORK(KMO_STA_TO_ACT_REO),1,2)
          WRITE(6,*) ' Orbitals reordered to standard order '
          IIFORM = 1
        END IF
*. Check subshells for supersymmetry
        IF(I_USE_SUPSYM.EQ.1) THEN
          IF(IPRORB.GE.5) THEN
            WRITE(6,*) ' Orbitals in shell form after MOROT'
            CALL PRINT_CMO_AS_SHELLS(WORK(KMOAOUT),IIFORM)
          END IF
*. Analyze eqivalence of components
          CALL ANA_SUBSHELLS_CMO(WORK(KMOAOUT),IIFORM,XMAXDF,
     &         MAXIRRDF,MAXSHLDF,0)
        END IF
      END IF
*
* ===========================
* Integral transformation
* ===========================
      IF(ITRA_FI.EQ.1) THEN
        WRITE(6,*) ' I am going to call TRAINT '
*. Direct integrals to current MO-integrals (assumed complete)
        KINT2 = KINT_2EMO
        KKCMO_I = KMOMO
        KKCMO_J = KMOMO
        KKCMO_K = KMOMO
        KKCMO_L = KMOMO
*. Complete integral transformation: from KINT_2EMO to KINT_2EINI
        IE2LIST_A = IE2LIST_FULL
        IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_A))
        KINT2_FSAVE = KINT2_A(IE2ARR_F)
        KINT2_A(IE2ARR_F) = KINT_2EINI
*
        IOCOBTP_A = 2
        INTSM_A = 1
        CALL TRAINT
C?      WRITE(6,*) ' WORK(KINT1), WORK(KINT2) = ',
C?   &               WORK(KINT1), WORK(KINT2)
*. Dump integrals from KINT1, KINT2 to file LU90 
        LU90 = 90
        KINT2 =  KINT_2EINI
        CALL DMPINT(LU90)
      END IF
*. And the one-electron file - is always written
      IF(NOMOFL.EQ.0) KMOAO = KMOAOUT
      IF(IPRORB.GE.1) THEN
        WRITE(6,*) ' Final, final, final orbitals: '
        CALL PRINT_CMOAO(WORK(KMOAOUT))
      END IF
*
      CALL DUMP_1EL_INFO(LU91)
*. Print info on the results(energies of the various calculations
      CALL E_SUMMARY
*. Print info on matrix multiplier
      CALL PR_MATML_STAT
*. Print info on  counters for kernel routines 
      CALL KERNEL_ROU_STAT_PRINT
*
      CALL QEXIT('REST ')
      CALL QSTAT
      IDUM=0
      CALL MEMMAN(IDUM,IDUM,'STATI',IDUM,'END   ')
C?    CALL RELEASE_WRKSPC()
      STOP ' I am home from  the loops '
      END
      SUBROUTINE MATCAS(CIN,COUT,NROWI,NROWO,IROWO1,NGCOL,ISCA,SCASGN)
*
* COUT(IR+IROWO1-1,ISCA(IC)) =
* COUT(IR+IROWO1-1,ISCA(IC)) + CIN(IR,IC)*SCASGN(IC)
* (if IGAT(IC).ne.0)
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION CIN(NROWI,*),COUT(NROWO,*)
      INTEGER ISCA(*)
      DIMENSION SCASGN(*)
*
      INCLUDE 'rou_stat.inc'
*
      NCALL_MATCAS = NCALL_MATCAS + 1
      MAXCOL = 0
      DO 100 IC = 1, NGCOL
        IF(ISCA(IC).NE.0) THEN
          ICEXP = ISCA(IC)
          MAXCOL = MAX(MAXCOL,ICEXP)
          SIGN = SCASGN(IC)
          DO 50 IR = 1,NROWI
            COUT(IR+IROWO1-1,ICEXP) =
     &      COUT(IR+IROWO1-1,ICEXP) + SIGN*CIN(IR,IC)
   50     CONTINUE
          XOP_MATCAS = XOP_MATCAS+FLOAT(NROWI)
        END IF
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Output from MATCAS '
        CALL WRTMAT(COUT,NROWO,MAXCOL,NROWO,MAXCOL)
      END IF
*
      RETURN
      END
C                 CALL MATCG(C,CB(ICGOFF),NROW,NIBTC,IBOT,
C                            NKBTC,I1,XI1S)
      SUBROUTINE MATCG(CIN,COUT,NROWI,NROWO,IROWI1,NGCOL,IGAT,GATSGN)
*
* Gather columns of CIN with phase
*
* COUT(IR,IC) = GATSGN(IC)*CIN(IR+IROWI1-1,IGAT(IC)) if IGAT(IC) .ne.0
* COUT(IR,IC) = 0                           if IGAT(IC) .ne.0
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER IGAT(*)
      DIMENSION GATSGN(*)
      DIMENSION CIN(NROWI,*),COUT(NROWO,*)
      INCLUDE 'rou_stat.inc'
*
      NCALL_MATCG = NCALL_MATCG + 1
C?    write(6,*) ' MATCG NROWI,NROWO,IROWI1,NGCOL '
C?    write(6,*)         NROWI,NROWO,IROWI1,NGCOL
      DO 100 IG = 1, NGCOL
C?      write(6,*) '  igat,sign ',IGAT(IG),GATSGN(IG)
        IF(IGAT(IG).EQ.0) THEN
          DO 20 IR = 1, NROWO
            COUT(IR,IG)=0.0D0
   20     CONTINUE
        ELSE
         IGFRM = IGAT(IG)
         SIGN = GATSGN(IG)
         DO 30 IR = 1, NROWO
C?         WRITE(6,'(A,4I3)') ' IR, IG, IROWI1, IGFRM ',
C?   &                          IR, IG, IROWI1, IGFRM
           COUT(IR,IG) = SIGN*CIN(IROWI1-1+IR,IGFRM)
   30    CONTINUE
         XOP_MATCG = XOP_MATCG + IR
        END IF
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Column gathered matrix '
        CALL WRTMAT(COUT,NROWO,NGCOL,NROWO,NGCOL)
      END IF
*
      RETURN
      END
      SUBROUTINE MATML5(C,A,B,NCROW,NCCOL,NAROW,NACOL,
     &                  NBROW,NBCOL,ITRNSP )
C
C MULTIPLY A AND B TO GIVE C
C
C     C = A * B             FOR ITRNSP = 0
C
C     C = A(TRANSPOSED) * B FOR ITRNSP = 1
C
C     C = A * B(TRANSPOSED) FOR ITRNSP = 2
C
C... JEPPE OLSEN, LAST REVISION JULY 24 1987
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION A(NAROW,NACOL),B(NBROW,NBCOL)
      DIMENSION C(NCROW,NCCOL)
*
CT    CALL QENTER('MATML')
*
      NTEST = 0
      IF ( NTEST .NE. 0 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' A AND B MATRIX FROM MATML5 '
        WRITE(6,*)
        CALL WRTMAT(A,NAROW,NACOL,NAROW,NACOL)
        CALL WRTMAT(B,NBROW,NBCOL,NBROW,NBCOL)
        WRITE(6,*)      ' NCROW NCCOL NAROW NACOL NBROW NBCOL '
        WRITE(6,'(6I6)')  NCROW,NCCOL,NAROW,NACOL,NBROW,NBCOL
        WRITE(6,*) ' ITRNSP: ', ITRNSP
      END IF
*. 
      IF(NAROW*NACOL*NBROW*NBCOL*NCROW*NCCOL .EQ. 0 ) THEN
        IZERO = 1
      ELSE
        IZERO = 0
      END IF
*
      IESSL = 0
      ICONVEX = 0
      IF(IESSL .EQ.1 .AND. IZERO .EQ. 0 ) THEN
*. Use IBM ESSL routines
*. I have been having a problem when all matrices and b are zero so
         IF(ITRNSP.EQ.0) THEN
         CALL DGEMUL(A,NAROW,'N',B,NBROW,'N',C,NCROW,NCROW,NACOL,NCCOL)
         ELSE IF (ITRNSP .EQ. 1 ) THEN
         CALL DGEMUL(A,NAROW,'T',B,NBROW,'N',C,NCROW,NCROW,NAROW,NCCOL)
         ELSE IF (ITRNSP .EQ. 2 ) THEN
         CALL DGEMUL(A,NAROW,'N',B,NBROW,'T',C,NCROW,NCROW,NACOL,NCCOL)
         ELSE 
           WRITE(6,*) ' Sorry MATML5 cannot follow your suggestion '
           WRITE(6,*) ' Since you suggest ITRNSP = ', ITRNSP
           WRITE(6,*) ' Please reprogram me ! '
         END IF
      ELSE IF (ICONVEX.EQ.1 ) THEN
*. DGEMM from CONVEX lib
	 LDA = MAX(1,NAROW)
	 LDB = MAX(1,NBROW)
	 LDC = MAX(1,NCROW)
	 IF(ITRNSP.EQ.0) THEN
	 CALL DGEMM('N','N',NAROW,NBCOL,NACOL,1.0D0,A,LDA,
     &               B,LDB,0.0D0,C,LDC)
	 ELSE IF (ITRNSP.EQ.1) THEN
	 CALL DGEMM('T','N',NACOL,NBCOL,NAROW,1.0D0,A,LDA,
     &               B,LDB,0.0D0,C,LDC)
	 ELSE IF(ITRNSP.EQ.2) THEN
	 CALL DGEMM('N','T',NAROW,NBROW,NACOL,1.0D0,A,LDA,
     &               B,LDB,0.0D0,C,LDC)
	 END IF
      ELSE     
* Use Jeppes version ( it should be working )
      IF( ITRNSP .NE. 0 ) GOTO 001
* ======
* C=A*B
* ======
        DO 30 J = 1, NCCOL
          DO 40 I = 1, NCROW
            T = 0.0D0
            DO 50 K = 1, NBROW
              T = T  + A(I,K)*B(K,J)
  50        CONTINUE
            C(I,J) = T
  40      CONTINUE
  30    CONTINUE
*
  001 CONTINUE
C
      IF ( ITRNSP .NE. 1 ) GOTO 101
* =========
* C=A(T)*B
* =========
        CALL SETVEC(C,0.0D0,NCROW*NCCOL)
        DO 150 J = 1, NCCOL
          DO 140 K = 1, NBROW
            DO 130 I = 1, NCROW
              C(I,J)= C(I,J) + A(K,I)*B(K,J)
  130       CONTINUE
  140     CONTINUE
  150   CONTINUE
C
  101 CONTINUE
C
      IF ( ITRNSP .NE. 2 ) GOTO 201
* ===========
*. C = A*B(T)
* ===========
        DO 250 J = 1,NCCOL
          DO 230 I = 1, NCROW
            T = 0.0D0
            DO 240 K = 1,NBCOL
              T  = T  + A(I,K)*B(J,K)
 240        CONTINUE
          C(I,J) = T
 230      CONTINUE
 250    CONTINUE
C
C
  201 CONTINUE
      END IF
C
      IF ( NTEST .NE. 0 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' C MATRIX FROM MATML5 '
        WRITE(6,*)
        CALL WRTMAT(C,NCROW,NCCOL,NCROW,NCCOL)
      END IF
C
CT    CALL QEXIT('MATML')
      RETURN
      END
C                MATSM2(SB,SB,CB,NIA,NIB,2)
      SUBROUTINE MATSM2(APB,A,B,NR,NC,ITP)
*
* ITP = 0: APB = A + B
* ITP = 1: APB = A + B Transposed
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      DIMENSION A(*),B(*)
*.Output
      DIMENSION APB(*)
*
      IF(ITP.EQ.0) THEN
        CALL VECSUM(APB,A,B,1.0D0,1.0D0,NR*NC)
      ELSE IF(ITP.EQ.1) THEN
        DO 100 IC = 1, NC
          DO  50 IR = 1, NR
            APB(IR+(IC-1)*NR) = A(IR+(IC-1)*NR)+B(IC+(IR-1)*NC)
   50     CONTINUE
  100   CONTINUE
      ELSE
        WRITE(6,*) ' MATSM2: Illegal transpose parameter '
        STOP 11
      END IF
*
      RETURN
      END
      FUNCTION MAXOCC(STRING,NEL,NSTRIN,NINOB,ISCR,NORB,NSTPSM,NSTSM)
*
* LARGEST NUMBER OF TIMES A GIVEN ELECTRON OCCURS
* IN STRINGS ( INACTIVE ORBITALS EXCLUDED )
*
      IMPLICIT REAL*8           (A-H,O-Z)
      INTEGER STRING(NEL,NSTRIN),ISCR(*),NSTPSM(*)
* ISCR SHOULD AT LEAST BE OF THE LENGTH NORB
*
      IMAX = 0
*. Eliminate Compiler warning
      IBASE = 1
      DO 200 ISTSM = 1, NSTSM
        CALL ISETVC(ISCR,0,NORB)
        IF(ISTSM.EQ.1) THEN
          IBASE = 1
        ELSE
          IBASE = IBASE+ NSTPSM(ISTSM-1)
        END IF
*
        DO 100 ISTRIN = IBASE,IBASE + NSTPSM(ISTSM) - 1
          DO 50 IEL = 1, NEL
            IF(STRING(IEL,ISTRIN).GT.NINOB)
     &      ISCR(STRING(IEL,ISTRIN) ) = ISCR(STRING(IEL,ISTRIN))+1
   50    CONTINUE
  100   CONTINUE
*
        DO 150 IORB = 1, NORB
         IMAX = MAX(IMAX,ISCR(IORB))
  150   CONTINUE
  200 CONTINUE
*
      NTEST = 1
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' LARGEST NUMBER OF TIMES A GIVEN ELECTRON OCCURS',
     &  IMAX
      END IF
      MAXOCC = IMAX
*
      RETURN
      END
      SUBROUTINE MEMCHK2(IDENT)
*
* Check memory allocated  with the memory manager
*
      INCLUDE 'implicit.inc'
      CHARACTER*6 IDENT
      CALL MEMMAN(IDUM,IDUM,'CHECK ',IDUM,IDENT)
*
      RETURN
      END
      SUBROUTINE MEMCHK
*
* Check memory allocated  with the memory manager
*
      CALL MEMMAN(IDUM,IDUM,'CHECK ',IDUM,'IDUM  ')
*
      RETURN
      END
      SUBROUTINE MEMMAN_ORG(KBASE,KADD,TASK,IR,IDENT)
*
* Memory manager routine
*
* Last modification; Oct. 15, 2012; Jeppe Olsen, modified output from for FREE
*
* KBASE: New base address
*         If TASK = INI, KBASE is offset for memory to be controlled
*         by MEMMAN
* KADD : Dimension of array to be added
*         If TASK = INI, KADD is total length of array
* TASK : = INI : Initialize                 Character*6
*         = ADDS : Add static memory
*         = ADDL : Add Local memory
*         = FLUSH : Flush local memory
*         = CHECK : Check memory paddings
*         = FREE  : Return first Free word in KBASE, and amoung of memory
*                   in KADD
*         = MARK  : Set a mark at current free adress
*         = FLUSM : Flush local memory to previous mark
*         = PRINT : Print memory map
* IR    : 1 => integer , 2 => real, 
*         ratio between integer and real is IRAT
* IDENT : identifier of memory slice,Character*6
*  
* Local Memory not flushed before allocation of additional static memory
* is tranferred to static memory
*  
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      CHARACTER*6 TASK,IDENT,IIDENT,MARKC,MAX_BLKC,MARK_ACT
*
      PARAMETER(NPAD = 1 )
      PARAMETER(MAXLVL = 10000)
      PARAMETER(MAXMRK = 10000)
*
      COMMON/CMEMO/NWORD,KFREES,KFREEL,NS,NL,NM,IBASE(MAXLVL),
     &             LENGTH(MAXLVL),IIDENT(MAXLVL),IMARK(MAXMRK),
     &             MARKL(MAXMRK),MARKS(MAXMRK),MARKC(MAXMRK),
     &             MAX_MEM,MAX_BLK,MAX_BLKC,MARK_ACT, KFREELT
*. Two real*8 words, one added NPAD times before each array, another
*. added NPAD times after each array
      DATA PAD1/0.123456789D0/
      DATA PAD2/0.987654321D0/
*
      INCLUDE 'irat.inc'
*. Info from matml7  
      COMMON/MATMLST/XNFLOP,XNCALL,XLCROW,XLCCOL,XLCROWCOL,TMULT
*. Info from copvec
      COMMON/COPVECST/XNCALL_COPVEC, XNMOVE_COPVEC
*
      INTEGER*8 IMEM
*
      ISTOP = 0                                                         
      ITSOK  = 1 
      IPRNTMP = 0
*
      IF(TASK(1:3).EQ.'INI') THEN
*
**.Initialize
*
        NS = 0
        NL = 0
        NSNLI = 0
        NM = 0
        KFREES = KBASE
        KFREEL = KBASE
        KFREELT = KBASE
*. KFREELT is free memory from last memory allocation
        NWORD = MAXMEM
        IPRNTMP = 0
        ISTOP = 0
*. Initialize info from matml7
        XNFLOP = 0.0D0
        XNCALL = 0.0D0
        XLCROW = 0.0D0
        XLCCOL = 0.0D0
        XLCROWCOL = 0.0D0
        TMULT     = 0.0D0
*. Initialize info from copvec
        XNCALL_COPVEC = 0
        XNMOVE_COPVEC = 0
*. Info on memory usage
        MAX_MEM = 0
        MAX_BLK = 0
*
** Return first free word in KBASE
*
      ELSE IF (TASK(1:4).EQ.'FREE') THEN
       KBASE = KFREEL
       NSNLI = NS+NL
       KADD = MXPWRD-KBASE + 1
*
**. Static memory
*
      ELSE IF(TASK(1:4).EQ.'ADDS') THEN
        KBASE = KFREEL+NPAD
        IF( IR .EQ. 1 ) THEN
          KFREES = KFREEL +(KADD+1)/IRAT + 2*NPAD
          IF((KADD+1)/IRAT.GT.MAX_BLK) THEN
            MAX_BLK  = (KADD+1)/IRAT
            MAX_BLKC = IDENT
          END IF
        ELSE
          KFREES = KFREEL + KADD + 2*NPAD
          IF(KADD.GT.MAX_BLK) THEN
            MAX_BLK  = KADD
            MAX_BLKC = IDENT
          END IF
        END IF
        IF ( KFREES-1 .GT. NWORD ) THEN
          WRITE(6,*)
          WRITE(6,*) ' You can''t always get what you want'
          WRITE(6,*) ' No, you can''t always get what you want'
          WRITE(6,*) ' But if you try sometime, you may find '
          WRITE(6,*) ' you get what you need '
          WRITE(6,*) '                       Jagger/Richard '
*
          WRITE(6,*) ' MEMMAN : work array too short '
          WRITE(6,*) ' current and required length ',NWORD,KFREES-1
*
          WRITE(6,*) ' Trying to allocate : identifer,length' 
          WRITE(6,'(20X,A,I12)')IDENT,KADD
          ISTOP = 1
          IPRNTMP = 1
          NSNLI = NS+NL
          GOTO 1001
        END IF
        NS = NS + NL + 1
        NL = 0
        NSNLI = NS + NL
        IF(NS.GT.MAXLVL) THEN
          WRITE(6,*) ' Too many levels in MEMMAN '
          WRITE(6,*) ' Increase MAXLVL from ', MAXLVL
          STOP 11
        END IF
        IIDENT(NS) = IDENT
        KFREEL = KFREES
        KFREELT = KFREEL
        IBASE(NS) = KBASE
C
C     linux: may crash here already at first ADD call to MEMMAN
C            when too large arrays are requested; strange ....
C
        DO 10 IPAD = 1, NPAD
          WORK(KBASE-NPAD-1+IPAD) = PAD1
          WORK(KFREEL-NPAD-1+IPAD) =  PAD2
   10   CONTINUE
*
**. Local memory
*
      ELSE IF(TASK(1:4).EQ.'ADDL') THEN
        IF(KADD.LT.0) THEN
          WRITE(6,*) ' MEMMAN: Allocation of negative memory slice '
          WRITE(6,*) ' MEMMAN: KADD = ', KADD
          WRITE(6,*) ' MEMMAN: Allocate with nonnegative  integers'
          WRITE(6,*) '          use FLUSH or FLUSM to deallocate   '
          WRITE(6,*) ' Trying to allocate: identifier,offset,  length' 
          WRITE(6,'(24X,A,2I12)')IDENT,KBASE,KADD
          ITSOK = 0
        END IF
        KFREEL_OLD = KFREEL
        KBASE = KFREEL+NPAD
        IF( IR .EQ. 1 ) THEN
          KFREEL = KFREEL +(KADD+1)/IRAT + 2*NPAD
          IF((KADD+1)/IRAT.GT.MAX_BLK) THEN
            MAX_BLK  = (KADD+1)/IRAT
            MAX_BLKC = IDENT
          END IF
        ELSE
          KFREEL = KFREEL + KADD + 2*NPAD
          IF(KADD.GT.MAX_BLK) THEN
            MAX_BLK  = KADD
            MAX_BLKC = IDENT
          END IF
        END IF
*. Check for integer overflow 
        IF(KFREEL.LT.KFREEL_OLD) THEN
           WRITE(6,*) ' MEMMAN: Overflow of memory pointer KFREE '
           WRITE(6,*) '           KFREEL_OLD, KADD, KFREEL = ',
     &                            KFREEL_OLD, KADD, KFREEL
*. Enforce program to print memory map and stop
           ITSOK = 0
           GOTO 1001
        END IF
*
        IF ( KFREEL-1 .GT. NWORD ) THEN
          WRITE(6,*)
          WRITE(6,*) ' You can''t always get what you want'
          WRITE(6,*) ' No, you can''t always get what you want'
          WRITE(6,*) ' But if you try sometime, you may find '
          WRITE(6,*) ' you get what you need '
          WRITE(6,*) '                       Jagger/Richard '
          
          WRITE(6,*) ' MEMMAN: work array too short '
          WRITE(6,*) ' current and required length ',NWORD,KFREEL-1 
          WRITE(6,*) ' Trying to allocate: identifier,offset,  length' 
          WRITE(6,'(24X,A,2I12)')IDENT,KBASE,KADD
          ISTOP = 1
          IPRNTMP = 1
          NSNLI = NS+NL
*. Reset KFREEL to previous value
          KFREEL = KFREEL_OLD
          GOTO 1001
        END IF
        NL =  NL + 1
        NSNLI = NS+NL
        KFREELT = KFREEL
        IF(NS+NL.GT.MAXLVL) THEN
          WRITE(6,*) ' Too many levels in MEMMAN '
          WRITE(6,*) ' Increase MAXLVL from ', MAXLVL
          ISTOP = 1
          IPRNTMP = 1
          NSNLI = NS+NL
        END IF
        IIDENT(NS+NL) = IDENT
        IBASE(NS+NL) = KBASE
        DO 20 IPAD = 1, NPAD
          WORK(KBASE-NPAD-1+IPAD) = PAD1
          WORK(KFREEL-NPAD-1+IPAD) =  PAD2
   20   CONTINUE
*
** Flush local memory
*
      ELSE IF(TASK(1:5).EQ.'FLUSH') THEN
        NSNLI = NS+NL
        KFREEL = KFREES
        NL = 0
*. Flush output unit
        LU6 = 6
        CALL GFLUSH(LU6)
      ELSE IF(TASK(1:4).EQ.'MARK') THEN
*. Set a mark at current free address
        NM = NM + 1
        IF(NM.GT.MAXMRK) THEN
          WRITE(6,*) ' Too many marks  in MEMMAN '
          WRITE(6,*) ' Increase MAXMRK from ', MAXMRK
          STOP 11
        END IF
        MARKC(NM) = IDENT 
        MARK_ACT = IDENT
        IMARK(NM) = KFREEL 
        MARKL(NM) = NL
        MARKS(NM) = NS
        NSNLI = NS + NL
      ELSE IF (TASK(1:5).EQ.'FLUSM') THEN
*. Flush memory to current MARK and eliminate mark
        IF(IDENT(1:6).NE.MARK_ACT(1:6)) THEN
          WRITE(6,*) ' Error in Flushing:  MARKS not consistent '
          WRITE(6,'(A,A,3X,A)') 
     &    ' Actual MARK and MARK to be flushed ',IDENT,MARK_ACT
          ITSOK = 0
          GOTO 1001
        END IF
*
        NSNLI = NS+NL
        KFREEL = IMARK(NM)
*. KFREELT is not updated untill after memory check- that is its call
        IF(KFREES.GT.IMARK(NM)) KFREES = IMARK(NM)
        IF(NM.GT.1) MARK_ACT = MARKC(NM-1)
        NL = MARKL(NM)
        NS = MARKS(NM)
        NM = NM - 1
*. Flush output unit
        LU6 = 6
        CALL GFLUSH(LU6)
      ELSE IF( TASK(1:5).EQ.'CHECK') THEN
        NSNLI = NS+ NL
      ELSE IF( TASK(1:5).EQ.'PRINT') THEN
        NSNLI = NS+ NL
        IPRNTMP = 1
      ELSE IF( TASK(1:5).EQ.'STATI') THEN
        NSNLI = NS + NL
        WRITE(6,'(/,x,77("="))')
        WRITE(6,*) ' Memory statistics:'
        IF (NWORD.GT.10*1024*1024/8) THEN
          IMEM = (8_8*INT(NWORD,8)-1_8)/(1024_8*1024_8)+1_8
          WRITE(6,'(/,x,a,i12,a,i10,a)')
     &         '  Maximum available work-space:    ',
     &         NWORD,
     &       ' R*8 words (',IMEM,' Mbytes)'
        ELSE
          WRITE(6,'(/,x,a,i12,a,i10,a)')
     &         '  Maximum available work-space:    ',
     &         NWORD,
     &       ' R*8 words (',(8*NWORD-1)/(1024)+1,' kbytes)'
        END IF
        IF (MAX_MEM.GT.10*1024*1024/8) THEN ! are we in the Mbyte regime
          IMEM = (8_8*INT(MAX_MEM,8)-1_8)/(1024_8*1024_8)+1_8
          WRITE(6,'(x,a,i12,a,i10,a)')
     &         '  Maximum of allocated work-space: ',
     &         MAX_MEM,
     &       ' R*8 words (',IMEM,' Mbytes)'
          IMEM = (8_8*INT(MAX_BLK,8)-1_8)/(1024_8*1024_8)+1_8
          WRITE(6,'(x,a,i12,a,i10,a/)')
     &         '  Largest allocated block:         ',
     &         MAX_BLK,
     &       ' R*8 words (',IMEM,' Mbytes)'
        ELSE
          WRITE(6,'(x,a,i12,a,i10,a)')
     &         '  Maximum of allocated work-space: ',
     &         MAX_MEM,
     &       ' R*8 words (',(8*MAX_MEM-1)/1024+1,' kbytes)'
          WRITE(6,'(x,a,i12,a,i10,a)')
     &         '  Largest allocated block:         ',
     &         MAX_BLK,
     &       ' R*8 words (',(8*MAX_BLK-1)/1024+1,' kbytes)'
        END IF
        WRITE(6,'(x,a,a,/)')
     &         '  Identifier of largest allocated block: ',
     &         MAX_BLKC   
        WRITE(6,'(x,77("="))')
      ELSE 
          WRITE(6,'(A,A6)') ' MEMMAN: Unknown task parameter ',TASK
          WRITE(6,'(A,A6)') ' MEMMAN: Corresponding IDENT ', IDENT
          WRITE(6,*) ' Too confused to continue  '
          STOP 11
      END IF
      MAX_MEM = MAX(MAX_MEM,KFREEL-1)
*
**. Check paddings
*
      ICHECK = 1
      ITSOK = 1
      IF(TASK(1:5).EQ.'CHECK'.OR.ICHECK.EQ.1) THEN
        DO 100 IL = 1, NSNLI
          JBASE = IBASE(IL)
          IF(IL.NE.NSNLI) THEN
           JBASEN = IBASE(IL+1)
          ELSE
           JBASEN = KFREELT + 1
          END IF
          L1OK = 1
          L2OK = 1
          DO  IPAD = 1, NPAD
            IF(WORK(JBASE-NPAD-1+IPAD).NE.PAD1 .OR.
     &         WORK(JBASEN-2*NPAD-1+IPAD).NE.PAD2) THEN
               ITSOK = 0
               WRITE(6,*) ' Memory problem for: '
               WRITE(6,*) '   Level (IL) ',IL 
               IF(WORK(JBASE-NPAD-1+IPAD).NE.PAD1) L1OK = 0
               IF(WORK(JBASEN-2*NPAD-1+IPAD).NE.PAD2) L2OK = 0
               IF(L1OK.EQ.1.AND.L2OK.EQ.1) THEN
                WRITE(6,'(1H ,4X,A,I10,4X,A)')
     &          IIDENT(IL),IBASE(IL),'    OKAY     OKAY '
               ELSE IF(L1OK.EQ.1.AND.L2OK.EQ.0) THEN
                WRITE(6,'(1H ,4X,A,I10,4X,A)')
     &          IIDENT(IL),IBASE(IL),'    OKAY       -  '
               ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.1) THEN
                WRITE(6,'(1H ,4X,A,I10,4X,A)')
     &          IIDENT(IL),IBASE(IL),'     -       OKAY '
               ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.1) THEN
                  WRITE(6,'(1H ,4X,A,I10,4X,A)')
     &            IIDENT(IL),IBASE(IL),'     -       OKAY '
               ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.0) THEN
                WRITE(6,'(1H ,4X,A,I10,4X,A)')
     &          IIDENT(IL),IBASE(IL),'     -       -    '
               END IF
            END IF
          END DO
  100   CONTINUE
      END IF
 1001 CONTINUE
*
*
        IF(ITSOK.EQ.0.OR.IPRNTMP.NE.0) THEN
          WRITE(6,'(A,A)') ' Current task: ', TASK
          WRITE(6,'(A,A)') ' Identifier  : ', IDENT
          WRITE(6,*) ' NS, NL, NSNLI',NS,NL,NSNLI
          IF(ITSOK.EQ.0)
     &    WRITE(6,*) '  Sorry to say it , but memory is CORRUPTED '
          WRITE(6,*) '  Memory map: '
          WRITE(6,*)
     &         '  Identifier   Offset    Length   Pad1 okay Pad2 okay '
          WRITE(6,*)
     &         '  ========== ========== ========= ========= ========= '
          DO 200 IL = 1, NSNLI
            JBASE = IBASE(IL)
            IF(IL.NE.NSNLI) THEN
             JBASEN = IBASE(IL+1)
            ELSE
             JBASEN = KFREELT + 1
            END IF
            LEN = JBASEN - JBASE - 2*NPAD
            IF(IL.EQ.NSNLI) LEN = LEN - 2
C?        WRITE(6,*) ' TEST: IL, JBASE, JBASEN = ', IL, JBASE, JBASEN
            L1OK = 1
            L2OK = 1
            DO 40 IPAD = 1, NPAD
              IF(WORK(JBASE-NPAD-1+IPAD).NE.PAD1) L1OK = 0
   40       CONTINUE
            DO 50 IPAD = 1, NPAD
              IF(WORK(JBASEN-2*NPAD-1+IPAD).NE.PAD2) L2OK = 0
   50       CONTINUE
            IF(L1OK.EQ.1.AND.L2OK.EQ.1) THEN
               WRITE(6,'(1H ,4X,A,2I10,4X,A)')
     &         IIDENT(IL),IBASE(IL),LEN,'    OKAY     OKAY '
            ELSE IF(L1OK.EQ.1.AND.L2OK.EQ.0) THEN
               WRITE(6,'(1H ,4X,A,2I10,4X,A)')
     &         IIDENT(IL),IBASE(IL),LEN,'    OKAY      --  '
            ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.1) THEN
               WRITE(6,'(1H ,4X,A,2I10,4X,A)')
     &         IIDENT(IL),IBASE(IL),LEN,'     --      OKAY '
            ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.1) THEN
               WRITE(6,'(1H ,4X,A,2I10,4X,A)')
     &         IIDENT(IL),IBASE(IL),LEN,'     --      OKAY '
            ELSE IF(L1OK.EQ.0.AND.L2OK.EQ.0) THEN
               WRITE(6,'(1H ,4X,A,2I10,4X,A)')
     &         IIDENT(IL),IBASE(IL),LEN,'     --       --  '
            END IF
  200     CONTINUE
*
          KFREELT = KFREEL
*
* Marks
*
          WRITE(6,*)
          WRITE(6,*) '======='
          WRITE(6,*) ' Marks '   
          WRITE(6,*) '======='
          WRITE(6,*)
*
          WRITE(6,*) ' Identifier  Start of free memory '
          WRITE(6,*) ' ================================='
          DO JMARK = 1, NM
            WRITE(6,'(3X,A6,10X,I10)') MARKC(JMARK),IMARK(JMARK)
          END DO
*
c        IF(ITSOK.EQ.0) STOP' Error observed by  memory manager '
          IF(ITSOK.EQ.0) THEN
            CALL QTRACE
            STOP' Error observed by  memory manager '
          END IF
        END IF
*
c      IF(ISTOP.NE.0) STOP ' Error observed by  memory manager '
      IF(ISTOP.NE.0) THEN
        CALL QTRACE
        STOP ' Error observed by  memory manager '
      END IF
      RETURN
      END
      SUBROUTINE MEMMAN(KBASE,KADD,TASK,IR,IDENT)
*
* Memory manager routine
*
* Last modification; Oct. 15, 2012; Jeppe Olsen, modified output from for FREE
*
* KBASE: New base address
*         If TASK = INI, KBASE is offset for memory to be controlled
*         by MEMMAN
* KADD : Dimension of array to be added
*         If TASK = INI, KADD is total length of array
* TASK : = INI : Initialize                 Character*6
*         = ADDS : Add static memory
*         = ADDL : Add Local memory
*         = FLUSH : Flush local memory
*         = CHECK : Check memory paddings
*         = FREE  : Return first Free word in KBASE, and amoung of memory
*                   in KADD
*         = MARK  : Set a mark at current free adress
*         = FLUSM : Flush local memory to previous mark
*         = PRINT : Print memory map
* IR    : 1 => integer , 2 => real, 
*         ratio between integer and real is IRAT
* IDENT : identifier of memory slice,Character*6
*  
* Local Memory not flushed before allocation of additional static memory
* is tranferred to static memory
*  
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      CHARACTER*6 TASK,IDENT,IIDENT,MARKC,MAX_BLKC,MARK_ACT
*
      PARAMETER(NPAD = 1 )
      PARAMETER(MAXLVL = 10000)
      PARAMETER(MAXMRK = 10000)
*
      COMMON/CMEMO/NWORD,KFREES,KFREEL,NS,NL,NM,IBASEL(MAXLVL),
     &             IBASES(MAXLVL),LENGTH(MAXLVL),IIDENTS(MAXLVL),
     &             IIDENTL(MAXLVL),
     &             IMARK(MAXMRK),MARKL(MAXMRK),MARKS(MAXMRK),
     &             MARKC(MAXMRK),MAX_MEM,MAX_BLK,MAX_BLKC,MARK_ACT, 
     &             KFREELT
*. Two real*8 words, one added NPAD times before each array, another
*. added NPAD times after each array
      DATA PAD1/0.123456789D0/
      DATA PAD2/0.987654321D0/
*
      INCLUDE 'irat.inc'
*. Info from matml7  
      COMMON/MATMLST/XNFLOP,XNCALL,XLCROW,XLCCOL,XLCROWCOL,TMULT
*. Info from copvec
      COMMON/COPVECST/XNCALL_COPVEC, XNMOVE_COPVEC
*
      INTEGER*8 IMEM
      LOGICAL   success
*
      ISTOP = 0                                                         
*
      IF (TASK(1:4).EQ.'FREE') THEN
         KADD = ma_inquire_stack(mt_dbl)
*
**. Static memory
*
      ELSE IF(TASK(1:4).EQ.'ADDS') THEN
        if (ir.eq.1) then
           success=ma_alloc_get(mt_int,KADD,IDENT,LBASE,KBASE)
        else
           success=ma_alloc_get(mt_dbl,KADD,IDENT,LBASE,KBASE)
        endif
        if (.not. success) then
          WRITE(6,*) ' Error allocating: identifier,offset,  length'
          WRITE(6,'(22X,A,2I12)')IDENT,KBASE,KADD
          call errquit('MEMMAN: ma alloc failed', KADD, MA_ERR)
        endif

        IF(NS.GT.MAXLVL) THEN
          WRITE(6,*) ' Too many levels in MEMMAN '
          WRITE(6,*) ' Increase MAXLVL from ', MAXLVL
          call errquit('MEMMAN: ma alloc failed', 911, MA_ERR)
        END IF
        NS = NS + 1
        IIDENTS(NS) = IDENT
        IBASES(NS) = LBASE
*
**. Local memory
*
      ELSE IF(TASK(1:4).EQ.'ADDL') THEN
        if (ir.eq.1) then
           success=ma_push_get(mt_int,KADD,IDENT,LBASE,KBASE)
        else 
           success=ma_push_get(mt_dbl,KADD,IDENT,LBASE,KBASE)
        endif
        if (.not. success) then
          WRITE(6,*) ' Error allocating: identifier,offset,  length'
          WRITE(6,'(22X,A,2I12)')IDENT,KBASE,KADD
          call errquit('MEMMAN: ma push failed', KADD, MA_ERR)
        endif

        NL =  NL + 1
        IF(NS.GT.MAXLVL) THEN
          WRITE(6,*) ' Too many levels in MEMMAN '
          WRITE(6,*) ' Increase MAXLVL from ', MAXLVL
          call errquit('MEMMAN: ma alloc failed', 911, MA_ERR)
        END IF
        IIDENTL(NL) = IDENT
        IBASEL(NL) = LBASE
*
** Flush local memory
*
      ELSE IF(TASK(1:5).EQ.'FLUSH') THEN
        if (.not. ma_chop_stack(IBASEL(1))) call
     $      errquit('MEMMAN: ma flush failed', 911, MA_ERR)
        NL = 0
*. Flush output unit
        LU6 = 6
        CALL GFLUSH(LU6)
      ELSE IF(TASK(1:4).EQ.'MARK') THEN
*. Set a mark at current free address
        NM = NM + 1
        IF(NM.GT.MAXMRK) THEN
          WRITE(6,*) ' Too many marks  in MEMMAN '
          WRITE(6,*) ' Increase MAXMRK from ', MAXMRK
          call errquit('MEMMAN: ma alloc failed', 911, MA_ERR)
        END IF
        MARKC(NM) = IDENT 
        MARK_ACT = IDENT
        IMARK(NM) = LBASE  
        MARKL(NM) = NL
        MARKS(NM) = NS
      ELSE IF (TASK(1:5).EQ.'FLUSM') THEN
*. Flush memory to current MARK and eliminate mark
        IF(IDENT(1:6).NE.MARK_ACT(1:6)) THEN
          WRITE(6,*) ' Error in Flushing:  MARKS not consistent '
          WRITE(6,'(A,A,3X,A)') 
     &    ' Actual MARK and MARK to be flushed ',IDENT,MARK_ACT
          call errquit('MEMMAN: ma alloc failed', 911, MA_ERR)
        END IF
*
        if (NL.gt.MARKL(NM)) then
           if (.not. ma_chop_stack(IBASEL(MARKL(NM)+1))) call
     $      errquit('MEMMAN: ma heap flush mark failed', 911, MA_ERR)
        endif
        if (NS.gt.MARKS(NM)) then
           if (.not. ma_chop_stack(IBASES(MARKS(NM)+1))) call
     $      errquit('MEMMAN: ma heap flush mark failed', 911, MA_ERR)
        endif
        NL = MARKL(NM)
        NS = MARKS(NM)
        NM = NM - 1
*. Flush output unit
        LU6 = 6
        CALL GFLUSH(LU6)
      ELSE IF( TASK(1:5).EQ.'CHECK') THEN
        if (.not. ma_verify_allocator_stuff()) call 
     &      errquit('MEMMAN: ma verify failed',911, MA_ERR)
      ELSE IF( TASK(1:5).EQ.'PRINT') THEN
        call ma_summarize_allocated_blocks()
      ELSE IF( TASK(1:5).EQ.'STATI') THEN
        call ma_print_stats(.true.)
      ELSE 
          WRITE(6,'(A,A6)') ' MEMMAN: Unknown task parameter ',TASK
          WRITE(6,'(A,A6)') ' MEMMAN: Corresponding IDENT ', IDENT
          WRITE(6,*) ' Too confused to continue  '
          call errquit('MEMMAN: ma alloc failed', 911, MA_ERR)
      END IF
      RETURN
      END
      Subroutine MkLundIO
************************************************************************
*                                                                      *
*     Purpose:                                                         *
*     Initialize the Common /LundIO/                                   *
*                                                                      *
*     Calling parameters: none                                         *
*                                                                      *
***** M.P. Fuelscher, University of Lund, Sweden, 1991 *****************
*
      Parameter ( mxBatch = 106  )
      Parameter ( mxSyBlk = 666  )
      Parameter ( lBlk    = 9600 )
      Parameter ( LuTwo   = 13   )
      Common / LundIO / LuTr2,lTr2Rec,iDAdr(mxBatch),nBatch(mxSyBlk)
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
      INTEGER*8 iToc,nBas,nOrb,nFro,nDel,Nsym
      INTEGER*8 iDAdr
      iDisk=0
 
*----------------------------------------------------------------------*
*     Start procedure:                                                 *
*     First set the unit number, record length and open file           *
*     per symmetry element                                             *
*----------------------------------------------------------------------*

      LuTr2=LuTwo
      lTr2Rec=lBlk
      Call DaName2(luTr2,'TRAINT')
cGLM      Call DaName(luTr2,'TRAINT')
*----------------------------------------------------------------------*
*     Load the table of disk adresses                                  *
*----------------------------------------------------------------------*
      Call dDaFile(LuTr2,2,iDAdr,mxBatch,iDisk)
cGLM      write(6,*) 'Am I here writing iDArd:'	  
cGLM	  write(6,*) (iDAdr(i),i=1,mxBatch)
cGLM      Call DaFile(LuTr2,2,iDAdr,mxBatch,iDisk)
*----------------------------------------------------------------------*
*     Generate the symmetry block to batch number translation table    *
*----------------------------------------------------------------------*
      iBatch=0
      Do iSym=1,8
        Do jSym=1,iSym
          Do kSym=1,iSym
            mxlSym=kSym
            If ( kSym.eq.iSym ) mxlSym=jSym
            Do lSym=1,mxlSym
              If ( ieor(iSym-1,jSym-1).eq.ieor(kSym-1,lSym-1) ) Then
                ijPair=jSym+iSym*(iSym-1)/2
                klPair=lSym+kSym*(kSym-1)/2
                iSyBlk=klPair+ijPair*(ijPair-1)/2
                iBatch=iBatch+1
                nBatch(iSyBlk)=iBatch
              End If
            End Do
          End Do
        End Do
      End Do
*----------------------------------------------------------------------*
*     Terminate procedure                                              *
*----------------------------------------------------------------------*
      Return
      End
      SUBROUTINE MLSM(IML,IPARI,ISM,TYPE,IWAY)
*
* Transfer between ML,IPARI notation and compound notation ISM
*
* IWAY = 1: IML,IPARI => Compound
* IWAY = 2: IML,IPARI <= Compound
*
* TYPE: 'SX','OB','ST','DX','CI'
*
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE 'mxpdim.inc'
      CHARACTER*2 TYPE
*./NONAB/
      LOGICAL INVCNT
      COMMON/NONAB/ INVCNT,NIG,NORASM(MXPOBS),
     &              MNMLOB,MXMLOB,NMLOB,
     &              MXMLST,MNMLST,NMLST,
     &              NMLSX ,MNMLSX,MXMLSX,
     &              MNMLCI,MXMLCI,NMLCI,
     &              MXMLDX,MNMLDX,NMLDX
*./CSM/
C     COMMON/CSM/NSMSX,NSMDX,NSMST,NSMCI,ITSSX,ITSDX
      INCLUDE 'csm.inc'
*
*.(Tired of warnings from 3090 Compiler so )
* (
      NML = 0
      MXML= 0
      MNML= 0
*             )
      IF(TYPE.EQ.'OB') THEN
        NML = NMLOB
        MXML = MXMLOB
        MNML = MNMLOB
      ELSE IF(TYPE.EQ.'SX') THEN
        NML = NMLSX
        MXML = MXMLSX
        MNML = MNMLSX
      ELSE IF(TYPE.EQ.'DX') THEN
        NML = NMLDX
        MXML = MXMLDX
        MNML = MNMLDX
      ELSE IF(TYPE.EQ.'ST') THEN
        NML = NMLST
        MXML = MXMLST
        MNML = MNMLST
      ELSE IF(TYPE.EQ.'CI') THEN
        NML = NMLCI
        MXML = MXMLCI
        MNML = MNMLCI
      END IF
*
      IF(IWAY.EQ.1) THEN
C        ISM = (IPARI-1)*NML + MNML - 1
         ISM = (IPARI-1)*NML + IML - MNML + 1
      ELSE IF(IWAY.EQ.2) THEN
        IF(ISM.GT.NML) THEN
          IPARI = 2
          IML = ISM - NML + MNML - 1
        ELSE
          IPARI = 1
          IML = ISM       + MNML - 1
        END IF
      ELSE
        WRITE(6,*) ' Error in MLSM , IWAY = ' ,IWAY
        WRITE(6,*) ' MLSM stop !!! '
        STOP 20
      END IF
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,'(A,A)') ' MLSM speaking ,type= ',TYPE
        WRITE(6,'(A,3I4)') ' IML IPARI ISM ',IML,IPARI,ISM
      END IF
*
      RETURN
      END
      SUBROUTINE MV7(CB,HCB,LUC,LUHC,XDUM,YDUM)
*
* Outer routine for sigma vector generation
* GAS version 
*
* IF ICISTR.gt.1, then CB, HCB are two blocks holding a batch
* IF ICISTR .eq. 1, then CB, HCB are two vectors holding a vector over
* parameters. Parameters are CSF's if required
*
* IF CSF's are active (NOCSF = 0), then three vectors over SD's 
* must be available (KCOMVECX_SD, X = 1, 2, 3)
*
* Written in terms of RASG3/SBLOCK, May 1997
* Code modified for ICISTR = 1 + CSF-SD, Jan. 2012
*
* Last modification; Oct. 30, 2012; Jeppe Olsen; call to Z_BLKFO changed
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
*
* =====
*.Input
* =====
*
*.Definition of c and sigma
      INCLUDE 'cands.inc'
*
*./ORBINP/: NACOB used
      INCLUDE 'orbinp.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cecore.inc'
      INCLUDE 'spinfo.inc'
      INCLUDE 'mv7task.inc'
      COMMON/CMXCJ/MXCJ,MAXK1_MX,LSCMAX_MX
*. Two blocks of C or Sigma (for ICISTR .gt. 2)
      DIMENSION CB(*),HCB(*)
*. Two vectors of C or Sigma (for ICISTR = 1)
COLD  DIMENSION C(*),HC(*)
*
      CALL QENTER('MV7  ')
*
      NTEST = 000
      NTEST = MAX(NTEST, IPRCIX)
      IF(NTEST.GE.5.AND.NTEST.LT.10) THEN
       WRITE(6,*) ' MV7 entered '
      ELSE IF(NTEST.GE.10) THEN
        WRITE(6,*)
        WRITE(6,*) ' Information from MV7: '
        WRITE(6,*) ' ======================'
        WRITE(6,*)
        WRITE(6,'(A,A)') ' Task is ', CMV7TASK
      END IF
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Input vector to MV7 '
        WRITE(6,*) ' ==================='
        IF(ICISTR.GT.1) THEN
          CALL WRTVCD(CB,LUC,1,-1)
        ELSE
          CALL WRTMAT(CB,1,NCVAR,1,NCVAR)
        END IF
      END IF
C?    WRITE(6,*) ' Ecore (MV7) = ', ECORE
        
      IDUM = 0
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'MV7   ')
*. For the moment
      ICFIRST = 1
      ISFIRST = 1
*
      MAXK1_MX = 0
      LSCMAX_MX = 0
      IF(ISSPC.LE.NCMBSPC) THEN
        IATP = 1
        IBTP = 2
      ELSE
        IATP = IALTP_FOR_GAS(ISSPC)
        IBTP = IBETP_FOR_GAS(ISSPC)
      END IF
      IF(NTEST.GE.10) WRITE(6,*) ' MV7TEST: IATP, IBTP = ', IATP, IBTP
*
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
*. Offset for supergroups 
      IOCTPA = IBSPGPFTP(IATP)
      IOCTPB = IBSPGPFTP(IBTP)
*
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
*. Block for storing complete or partial CI-vector
      IF(ISIMSYM.EQ.0) THEN
        LBLOCK = MXSOOB
      ELSE
        LBLOCK = MXSOOB_AS
      END IF
*. Why the below, this is size of 'inner batch'
      IF(NOCSF.EQ.0.OR.ICNFBAT.EQ.-2) THEN
        LBLOCK  = MAX(NSD_FOR_OCCLS_MAX,MXSOOB)
      END IF
      LBLOCK = MAX(LBLOCK,LCSBLK)
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' TEST, MV7: LCSBLK, LBLOCK, MXSOOB  = ', 
     &                          LCSBLK, LBLOCK, MXSOOB
      END IF
      ICOMP = 0
      ILTEST = -3006
      CALL MEMCHK2('MV7BEZ')
C?    WRITE(6,*) ' KSIOIO = ', KSIOIO
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' ICSM, ISSM = ', ICSM, ISSM
        WRITE(6,*) ' KCLBT, KSLBT(a) == ', KCLBT, KSLBT
        WRITE(6,*) ' WORK(KCLBT): '
        CALL IWRTMA(int_mb(KCLBT),1,1,1,1)
      END IF
*
      IF(ISFIRST.EQ.1) THEN
        CALL Z_BLKFO_FOR_CISPACE(ISSPC,ISSM,LBLOCK,ICOMP,
     &       NTEST,NSBLOCK,NSBATCH,
     &       dbl_mb(KSIOIO),int_mb(KSBLTP),NSOCCLS_ACT,dbl_mb(KSIOCCLS_ACT),
     &       int_mb(KSLBT),int_mb(KSLEBT),int_mb(KSLBLK),int_mb(KSI1BT),
     &       int_mb(KSIBT),
     &       int_mb(KSNOCCLS_BAT),int_mb(KSIBOCCLS_BAT),ILTEST)
        NSOCCLS = NSOCCLS_ACT
      END IF
      IF(ICFIRST.EQ.1) THEN
        CALL Z_BLKFO_FOR_CISPACE(ICSPC,ICSM,LBLOCK,ICOMP,
     &       NTEST,NCBLOCK,NCBATCH,
     &       int_mb(KCIOIO),int_mb(KCBLTP),NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),
     &       int_mb(KCLBT),int_mb(KCLEBT),int_mb(KCLBLK),int_mb(KCI1BT),
     &       int_mb(KCIBT),
     &       int_mb(KCNOCCLS_BAT),int_mb(KCIBOCCLS_BAT),ILTEST)
        NCOCCLS = NCOCCLS_ACT
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' KCLBT, KSLBT(b) == ', KCLBT, KSLBT
          WRITE(6,*) ' WORK(KCLBT): '
          CALL IWRTMA(int_mb(KCLBT),1,1,1,1)
        END IF
      END IF
      CALL MEMCHK2('MV7AFZ')
C     WRITE(6,*) ' ECORE in MV7 =', ECORE
*. Number of BLOCKS
        NBLOCK = NSBLOCK
C?      WRITE(6,*) ' Number of blocks ', NBLOCK

      IF(I12.EQ.2) THEN
        IDOH2 = 1
      ELSE
        IDOH2 = 0
      END IF
*
      IF(NOCSF.EQ.0.AND.ICNFBAT.GE.2) THEN
*. Obtain scratch files for saving combination forms of C and Sigma
C             FILEMAN_MINI(IFILE,ITASK)
         CALL FILEMAN_MINI(LU_CDET,'ASSIGN')
         CALL FILEMAN_MINI(LU_SDET,'ASSIGN')
         IF(NTEST.GE.1000)  THEN
           WRITE(6,*) ' Test: LU_CDET, LU_SDET: ',
     &                        LU_CDET, LU_SDET
         END IF
* ITASK = ASSIGN => Find a free superscratchfile, reserve, set IFILE to 
* ITASK = FREE   => Free superscratchfile IFILE
      END IF
*
      IF(ICISTR.EQ.1) THEN
       LLUC = 0
       LLUHC = 0
      ELSE 
       IF(NOCSF.EQ.1) THEN
        LLUC = LUC
        LLUHC = LUHC
       ELSE
        LLUC = LU_CDET
        LLUHC = LU_SDET
       END IF
      END IF

      IF(NOCSF.EQ.0) THEN
       IF(ICNFBAT.EQ.1) THEN
*. In core
         CALL CSDTVCM(CB,WORK(KCOMVEC1_SD),WORK(KCOMVEC2_SD),
     &                1,0,ICSM,ICSPC,2)
       ELSE
*. Not in core- write determinant expansion on LU_CDET 
C       CSDTVCMN(CSFVEC,DETVEC,SCR,IWAY,ICOPY,ISYM,ISPC,
C    &           IMAXMIN_OR_GAS,ICNFBAT,LU_DET,LU_CSF,NOCCLS_ACT,
C    &           IOCCLS_ACT,IBLOCK,NBLK_PER_BATCH)  
        CALL CSDTVCMN(CB,HCB,WORK(KVEC3),
     &       1,0,ICSM,ICSPC,2,2,LU_CDET,LUC,NCOCCLS_ACT,
     &       dbl_mb(KCIOCCLS_ACT),int_mb(KCIBT),int_mb(KCLBT))
       END IF
      END IF
*
C            RASSG3(CB,SB,LBATS,LEBATS,I1BATS,IBATS,LUC,LUHC,C,HC,ECORE)
      IF(ICISTR.GE.2) THEN
        CALL RASSG3(CB,HCB,NSBATCH,int_mb(KSLBT),int_mb(KSLEBT),
     &       int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,XDUM,XDUM,ECORE,
     &       CMV7TASK)
      ELSE
*. ICISTR = 1, CB, HCB are the complete vectors
        IF(NOCSF.EQ.1) THEN
*. CB and HCB on input are the complete vectors
          CALL RASSG3(WORK(KVEC1P),WORK(KVEC2P),NSBATCH,
     &         int_mb(KSLBT),int_mb(KSLEBT),
     &         int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,CB,HCB,ECORE,
     &         CMV7TASK)
*. Input is in KCOMVEC1_SD, construct output in KCOMVEC2_SD
          IF(NTEST.GE.1000) THEN
            WRITE(6,*) ' NSVAR elements of output vector from RASSG3'
            CALL WRTMAT(WORK(KVEC2P),1,NSVAR,1,NSVAR)
          END IF
        ELSE
          CALL RASSG3(WORK(KVEC1P),WORK(KVEC2P),NSBATCH,
     &         int_mb(KSLBT),int_mb(KSLEBT),
     &         int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,
     &         WORK(KCOMVEC1_SD),WORK(KCOMVEC2_SD),ECORE,
     &        CMV7TASK)
          IF(NTEST.GE.1000) THEN
            WRITE(6,*) ' NSVAR elements of output vector from RASSG3'
            CALL WRTMAT(WORK(KCOMVEC2_SD),1,NSVAR,1,NSVAR)
          END IF
        END IF ! CSF switch
      END IF ! ICISTR switch
*
      IF(NOCSF.EQ.0) THEN
* Transform sigma vector in KCOMVEC2_SD to CSF basis
       IF(ICNFBAT.EQ.1) THEN
C CSDTVCM(CSFVEC,DETVEC,IWAY,ICOPY,ISYM,ISPC,IMAXMIN_OR_GAS)
         CALL CSDTVCM(HCB,WORK(KCOMVEC2_SD),WORK(KCOMVEC1_SD),
     &        2,0,ISSM,ISSPC,2)
       ELSE
        CALL CSDTVCMN(HCB,CB,WORK(KVEC3),
     &       2,0,ISSM,ISSPC,2,2,LU_SDET,LUHC,NSOCCLS_ACT,
     &       dbl_mb(KSIOCCLS_ACT),WORK(KSIBT),int_mb(KSLBT))
       END IF
      END IF
*
      IF(NOCSF.EQ.0.AND.ICNFBAT.GE.2) THEN
        CALL FILEMAN_MINI(LU_CDET,'FREE  ')
        CALL FILEMAN_MINI(LU_SDET,'FREE  ')
      END IF
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Output vector from MV7 '
        WRITE(6,*) ' ===================== '
        IF(ICISTR.GT.1) THEN
          CALL WRTVCD(CB,LUHC,1,-1)
        ELSE 
          CALL WRTMAT(HCB,1,NSVAR,1,NSVAR)
        END IF
      END IF
*
*. Eliminate local memory
      CALL MEMMAN(KDUM ,IDUM,'FLUSM ',2,'MV7   ')
*
      CALL QEXIT('MV7  ')
*
      RETURN
      END
      FUNCTION NDXFSM(NSMOB,NSMSX,MXPOBS,NO1PS,NO2PS,NO3PS,NO4PS,
     &         IDXSM,ADSXA,SXDXSX,IS12,IS34,IS1234,IPRNT)
*
* Number of double excitations with total symmetry IDXSM
*
* IS12 (0,1,-1)   : Permutational symmetry between index 1 and 2
* IS34 (0,1,-1)   : Permutational symmetry between index 3 and 3
* IS1234 (0,1,-1) : permutational symmetry between index 12 and 34
*
*. General input
      INTEGER ADSXA(MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
*. Specific input
      INTEGER NO1PS(*),NO2PS(*),NO3PS(*),NO4PS(*)
*
*
      N12 = 0
      N34 = 0
      MDX = 0
      DO 200 I12SM = 1, NSMSX
        DO 190 I1SM = 1, NSMOB
          I2SM = ADSXA(I1SM,I12SM)
          IF(IS12.NE.0.AND.I1SM.LT.I2SM) GOTO 190
          IF(IS12.EQ.0) THEN
           I12NUM = (I1SM-1)*NSMSX+I2SM
          ELSE
           I12NUM =  I1SM*(I1SM+1)/2+I2SM
          END IF
          IF(IS12.EQ.0.OR.I1SM.NE.I2SM) THEN
            N12 = NO1PS(I1SM)*NO2PS(I2SM)
          ELSE IF(IS12.EQ.1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
          ELSE IF(IS12.EQ.-1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
          END IF
          I34SM = SXDXSX(I12SM,IDXSM)
          DO 90 I3SM = 1, NSMOB
            I4SM = ADSXA(I3SM,I34SM)
            IF(IS34.NE.0.AND.I3SM.LT.I4SM) GOTO 90
            IF(IS34.EQ.0) THEN
             I34NUM = (I3SM-1)*NSMSX+I4SM
            ELSE
             I34NUM =  I3SM*(I3SM+1)/2+I4SM
            END IF
            IF(IS1234.NE.0.AND.I12NUM.LT.I34NUM) GOTO 90
            IF(IS34.EQ.0.OR.I3SM.NE.I4SM) THEN
            N34 = NO3PS(I3SM)*NO4PS(I4SM)
            ELSE IF(IS34.EQ.1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
            ELSE IF(IS34.EQ.-1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
            END IF
            IF(IS1234.EQ.0.OR.I12NUM.NE.I34NUM) THEN
              MDX = MDX + N12 * N34
            ELSE IF( IS1234.EQ.1.AND.I12NUM.EQ.I34NUM) THEN
              MDX =  MDX + N12*(N12+1)/2
              ELSE IF( IS1234.EQ.-1.AND.I12NUM.EQ.I34NUM) THEN
              MDX =  MDX + N12*(N12-1)/2
            END IF
C?          WRITE(6,*) ' I1SM I2SM I3SM I4SM MDX '
C?          WRITE(6,*)   I1SM,I2SM,I3SM,I4SM,MDX
   90       CONTINUE
C 100     CONTINUE
  190   CONTINUE
  200 CONTINUE
*
      NDXFSM = MDX
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
      IF(NTEST.NE.0) THEN
         WRITE(6,*) ' Number of double excitations obtained ', MDX
      END IF
*
      RETURN
      END
      FUNCTION NDXFSM2(NSMOB,NSMSX,MXPOBS,NO1PS,NO2PS,NO3PS,NO4PS,
     &         IDXSM,ADSXA,SXDXSX,IS12,IS34,IS1234,IPRNT)
*
* Number of double excitations with total symmetry IDXSM
*
* IS12 (0,1,-1)   : Permutational symmetry between index 1 and 2
* IS34 (0,1,-1)   : Permutational symmetry between index 3 and 3
* IS1234 (0,1,-1) : permutational symmetry between index 12 and 34
*
*. General input
      INTEGER ADSXA(MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
*. Specific input
      INTEGER NO1PS(*),NO2PS(*),NO3PS(*),NO4PS(*)
*
*
      MM = 0
      N12 = 0
      N34 = 0
      MDX = 0
      DO 200 I12SM = 1, NSMSX
        DO 190 I1SM = 1, NSMOB
          IF (NO1PS(I1SM).EQ.0) GOTO 190
          I2SM = ADSXA(I1SM,I12SM)
          IF (NO2PS(I2SM).EQ.0) GOTO 190
          IF(IS12.NE.0.AND.I1SM.LT.I2SM) GOTO 190
           I12NUM = (MIN(I1SM,I2SM)-1)*NSMOB+MAX(I1SM,I2SM)
          I12DIA = 0
          IF(IS12.EQ.0.OR.I1SM.NE.I2SM) THEN
            N12 = NO1PS(I1SM)*NO2PS(I2SM)
            I12DIA=0
          ELSE IF(IS12.EQ.1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
            I12DIA=1
          ELSE IF(IS12.EQ.-1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
            I12DIA=-1
          END IF
          I34SM = SXDXSX(I12SM,IDXSM)
          DO 90 I3SM = 1, NSMOB
            IF(NO3PS(I3SM).EQ.0) GOTO 90
            I4SM = ADSXA(I3SM,I34SM)
            IF(NO4PS(I4SM).EQ.0) GOTO 90
            IF(IS34.NE.0.AND.I3SM.LT.I4SM) GOTO 90
             I34NUM = (MIN(I3SM,I4SM)-1)*NSMOB+MAX(I3SM,I4SM)
            IF(IS1234.NE.0.AND.I12NUM.LT.I34NUM) GOTO 90
            I34DIA = 0
            IF(IS34.EQ.0.OR.I3SM.NE.I4SM) THEN
              N34 = NO3PS(I3SM)*NO4PS(I4SM)
              I34DIA=0
            ELSE IF(IS34.EQ.1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
              I34DIA=1
            ELSE IF(IS34.EQ.-1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
              I34DIA=-1
            END IF
            IF(I12DIA.EQ.-1.OR.I34DIA.EQ.-1) THEN
              WRITE(6,*) 'Implementation incomplete...'
              STOP 'NDXFSM2'
            END IF
            IF(IS1234.EQ.0.OR.I12NUM.NE.I34NUM) THEN
              IF (I12DIA.EQ.I34DIA) THEN
                MDX = MDX + N12 * N34
              ELSE IF (I12DIA.EQ.1.AND.I34DIA.EQ.0) THEN
                IF (I3SM.NE.I4SM) STOP 'non-covered exception'
                N12OOD=NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
                N12DIA=NO1PS(I1SM)
                N34LOW=NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
                MDX = MDX + N12OOD*N34 + N12DIA*N34LOW
              ELSE IF (I12DIA.EQ.0.AND.I34DIA.EQ.1) THEN
                IF (I1SM.NE.I2SM) STOP 'non-covered exception'
                N34OOD=NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
                N34DIA=NO3PS(I3SM)
                N12LOW=NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
                MDX = MDX + N34OOD*N12 + N34DIA*N12LOW
              ELSE
                STOP 'non-covered case'
              END IF
            ELSE IF (IS12.NE.0.AND.IS34.EQ.0.AND.IS1234.NE.0.AND.
     &               I1SM.EQ.I4SM.AND.I2SM.EQ.I3SM.AND.
     &               I12DIA.EQ.0.AND.I34DIA.EQ.0) THEN
              MDX = MDX + N12*(N12+1)/2
            ELSE IF( IS1234.EQ.1.AND.I12NUM.EQ.I34NUM) THEN
              IF (I12DIA.EQ.I34DIA) THEN
                MDX =  MDX + N12*(N12+1)/2
              ELSE
                ! additional elements
                ! I1SM==I2SM==I3SM==14SM, anyway, so:
                NN = (NO1PS(I1SM)-1)*NO1PS(I1SM)/2
                IF(ABS(I12DIA).EQ.1) MM=N12
                IF(ABS(I34DIA).EQ.1) MM=N34
                MDX = MDX + MM*(MM+1)/2 + NN*(NN+1)/2
              END IF
            ELSE IF( IS1234.EQ.-1.AND.I12NUM.EQ.I34NUM) THEN
              IF (I12DIA.EQ.I34DIA) THEN
                MDX =  MDX + N12*(N12-1)/2
              ELSE
                ! additional elements
                ! I1SM==I2SM==I3SM==14SM, anyway, so:
                NN = (NO1PS(I1SM)-1)*NO1PS(I1SM)/2
                IF(ABS(I12DIA).EQ.1) MM=N12
                IF(ABS(I34DIA).EQ.1) MM=N34
                MDX = MDX + MM*(MM-1)/2 + NN*(NN+1)/2
              END IF
            END IF
C?          WRITE(6,*) ' I1SM I2SM I3SM I4SM MDX '
C?          WRITE(6,*)   I1SM,I2SM,I3SM,I4SM,MDX
   90       CONTINUE
C 100     CONTINUE
  190   CONTINUE
  200 CONTINUE
*
      NDXFSM2 = MDX
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
      IF(NTEST.NE.0) THEN
         WRITE(6,*) ' Number of double excitations obtained ', MDX
      END IF
*
      RETURN
      END
      SUBROUTINE NEXREC(LU,LBLK,REC,IEND,LENGTH)
*
* OBTAIN NEXT RECORD ON FILE LU, IF
* AN END OF FILE IS ISSUED THE RECORD IS EMPTY
* AND IEND IS SET TO 1
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION REC(*)
*
      IF(LBLK .GT. 0 ) THEN
        IEND = 0
        LENGTH = LBLK
COLD    CALL FRMDSC(REC,LENGTH,LBLK,LU)
        CALL FRMDSC(REC,LENGTH,-1,LU,IMZERO,IAMPACK)
      ELSE
        CALL IFRMDS(LENGTH,1,LBLK,LU)
C?    write(6,*) ' Length in NEXREC ',LENGTH
        IF(LENGTH.GE.0) THEN
          IEND = 0
          CALL FRMDSC(REC,LENGTH,LBLK,LU,IMZERO,IAMPACK)
C?     write(6,*) ' Record read in '
C?     CALL WRTMAT(REC,1,LENGTH,1,LENGTH)
        ELSE
          IEND = 1
        END IF
      END IF
*
      RETURN
      END
      SUBROUTINE NSTRSO(NEL,NORB1,NORB2,NORB3,
     &                  NELMN1,NELMX1,NELMN3,NELMX3,
     &                  IOC,NORB,NSTASO,NOCTYP,NSMST,IOTYP,IPRNT)
*
* Number of strings per type and symmetry
*
* Jeppe Olsen Winter of 1990
*
      IMPLICIT REAL*8           ( A-H,O-Z)
      DIMENSION IOC(*),NSTASO(NOCTYP,NSMST)
*
      CALL ISETVC(NSTASO,0,NSMST*NOCTYP)
      NTEST0 = 0
      NTEST = MAX(IPRNT,NTEST0)
      NSTRIN = 0
      IORB1F = 1
      IORB1L = IORB1F+NORB1-1
      IORB2F = IORB1L + 1
      IORB2L = IORB2F+NORB2-1
      IORB3F = IORB2L + 1
      IORB3L = IORB3F+NORB3-1
* Loop over possible partitionings between RAS1,RAS2,RAS3
      DO 1001 IEL1 = NELMX1,NELMN1,-1
      DO 1003 IEL3 = NELMN3,NELMX3, 1
       IF(IEL1.GT. NORB1 ) GOTO 1001
       IF(IEL3.GT. NORB3 ) GOTO 1003
       IEL2 = NEL - IEL1-IEL3
       IF(IEL2 .LT. 0 .OR. IEL2 .GT. NORB2 ) GOTO 1003
       IFRST1 = 1
* Loop over RAS 1 occupancies
  901  CONTINUE
         IF( IEL1 .NE. 0 ) THEN
           IF(IFRST1.EQ.1) THEN
            CALL ISTVC2(IOC(1),0,1,IEL1)
            IFRST1 = 0
           ELSE
             CALL NXTORD(IOC,IEL1,IORB1F,IORB1L,NONEW1)
             IF(NONEW1 .EQ. 1 ) GOTO 1003
           END IF
         END IF
         IF( NTEST .GE.500) THEN
           WRITE(6,*) ' RAS 1 string '
           CALL IWRTMA(IOC,1,IEL1,1,IEL1)
         END IF
         IFRST2 = 1
         IFRST3 = 1
* Loop over RAS 2 occupancies
  902    CONTINUE
           IF( IEL2 .NE. 0 ) THEN
             IF(IFRST2.EQ.1) THEN
              CALL ISTVC2(IOC(IEL1+1),IORB2F-1,1,IEL2)
              IFRST2 = 0
             ELSE
               CALL NXTORD(IOC(IEL1+1),IEL2,IORB2F,IORB2L,NONEW2)
               IF(NONEW2 .EQ. 1 ) THEN
                 IF(IEL1 .NE. 0 ) GOTO 901
                 IF(IEL1 .EQ. 0 ) GOTO 1003
               END IF
             END IF
           END IF
           IF( NTEST .GE.500) THEN
             WRITE(6,*) ' RAS 1 2 string '
             CALL IWRTMA(IOC,1,IEL1+IEL2,1,IEL1+IEL2)
           END IF
           IFRST3 = 1
* Loop over RAS 3 occupancies
  903      CONTINUE
             IF( IEL3 .NE. 0 ) THEN
               IF(IFRST3.EQ.1) THEN
                CALL ISTVC2(IOC(IEL1+IEL2+1),IORB3F-1,1,IEL3)
                IFRST3 = 0
               ELSE
                 CALL NXTORD(IOC(IEL1+IEL2+1),
     &           IEL3,IORB3F,IORB3L,NONEW3)
                 IF(NONEW3 .EQ. 1 ) THEN
                   IF(IEL2 .NE. 0 ) GOTO 902
                   IF(IEL1 .NE. 0 ) GOTO 901
                   GOTO 1003
                 END IF
               END IF
             END IF
             IF( NTEST .GE. 500) THEN
               WRITE(6,*) ' RAS 1 2 3 string '
               CALL IWRTMA(IOC,1,NEL,1,NEL)
             END IF
* Next string has been constructed , Enlist it !.
             NSTRIN = NSTRIN + 1
*. Symmetry of string
             ISYM = ISYMST(IOC,NEL)
C                   ISYMST(STRING,NEL)
*. occupation type of string
             ITYP = IOCTP2(IOC,NEL,IOTYP)
C                   IOCTP2(STRING,NEL)
*
             NSTASO(ITYP,ISYM) = NSTASO(ITYP,ISYM)+ 1
*
           IF( IEL3 .NE. 0 ) GOTO 903
           IF( IEL3 .EQ. 0 .AND. IEL2 .NE. 0 ) GOTO 902
           IF( IEL3 .EQ. 0 .AND. IEL2 .EQ. 0 .AND. IEL1 .NE. 0)
     &     GOTO 901
 1003 CONTINUE
 1001 CONTINUE
*
 
      IF(NTEST .GT. 0 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' Number of strings generated   ', NSTRIN
        WRITE(6,*)
        WRITE(6,*) ' NUMBER OF STRINGS PER SYM(COL) AND TYPE ( ROW)'
        WRITE(6,*) '================================================'
        CALL IWRTMA(NSTASO,NOCTYP,NSMST,NOCTYP,NSMST)
      END IF
C
      RETURN
      END
      FUNCTION NSXFSM(NSMOB,MXPOBS,NO1PS,NO2PS,ISXSM,ADSXA,
     &ISYM,IPRNT)
*
* Number of single excitations of symmetry ISXSM
*
* ISYM = 0: All symmetry allowed excitations
* ISYM = 1: Only excitations a+iaj with I.ge.J
* ISYM =-1: Only excitations a+iaj with I.gt.J
      INTEGER ADSXA(MXPOBS,2*MXPOBS)
      INTEGER NO1PS(*),NO2PS(*)
*
      MSXFSM = 0
C?    WRITE(6,*) ' NSMOB ',NSMOB
      DO 100 IO1SM = 1,NSMOB
        IO2SM = ADSXA(IO1SM,ISXSM)
C?      WRITE(6,*) ' IO1SM,IO2SM',IO1SM,IO2SM
        IF(ISYM.EQ.0.OR.IO1SM.GT.IO2SM) THEN
          MSXFSM = MSXFSM + NO1PS(IO1SM)*NO2PS(IO2SM)
        ELSE IF( ISYM.EQ. 1 .AND. IO1SM.EQ.IO2SM) THEN
          MSXFSM = MSXFSM + NO1PS(IO1SM)*(NO1PS(IO1SM)+1)/2
        ELSE IF( ISYM.EQ.-1 .AND. IO1SM.EQ.IO2SM) THEN
          MSXFSM = MSXFSM + NO1PS(IO1SM)*(NO1PS(IO1SM)-1)/2
        END IF
  100 CONTINUE
*
      NSXFSM = MSXFSM
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
 
      IF(NTEST.NE.0) THEN
        WRITE(6,*)
     &  ' Number of single excitations of symmetry ',ISXSM,',',NSXFSM
      END IF
*
      RETURN
      END
      FUNCTION NUMST3(NEL,NORB1,NEL1MN,NEL1MX,NORB2,
     &                NORB3,NEL3MN,NEL3MX)
*
* Number of strings with NEL electrons that fullfills
*
* Between NEL1MN AND NEL1MX electrons in the first NORB1 orbitals
* Between NEL3MN AND NEL3MX electrons in the last  NORB3 orbitals
*
*
*
      NTEST = 0
      NSTRIN = 0
*
      DO 100 IEL1 = NEL1MN,MIN(NEL1MX,NORB1,NEL)
        NSTIN1 = IBION(NORB1,IEL1)
        IEL3MN = MAX ( NEL3MN,NEL-(IEL1+NORB2) )
        IEL3MX = MIN ( NEL3MX,NEL-IEL1)
        DO 80 IEL3 = IEL3MN, IEL3MX
         IEL2 = NEL - IEL1-IEL3
         NSTINT = NSTIN1*IBION(NORB2,IEL2)*IBION(NORB3,IEL3)
         NSTRIN = NSTRIN + NSTINT
  80   CONTINUE
 100  CONTINUE
      NUMST3 = NSTRIN
*
      IF( NTEST .GE.1 )
     &WRITE(6,'(/A,I6)') '  Number of strings generated ... ', NSTRIN
*
      RETURN
      END
      SUBROUTINE NXTBLK(IATP,IBTP,IASM,NOCTPA,NOCTPB,NSMST,IBLTP,IDC,
     &                  NONEW,IOCOC,ISMOST,
     &                  NSSOA,NSSOB,LBLOCK,LBLOCKP)
*
* Obtain allowed block following IATP IBTP IASM
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER IBLTP(*),ISMOST(NSMST)
      INTEGER NSSOA(NSMST,*),NSSOB(NSMST,*)
      INTEGER IOCOC(NOCTPA,NOCTPB)
*
*.Initialize
*
      ISM = IASM
      IA = IATP
      IB = IBTP
      NONEW = 0
*. Loop over blocks in batch
 1000 CONTINUE
*.  New order : ISM,IB,IA (leftmost inner loop )
      IF(ISM.LT.NSMST) THEN
        ISM = ISM + 1
      ELSE
        ISM = 1
        IF(IB.LT.NOCTPB) THEN
          IB = IB + 1
        ELSE
          IB = 1
          IF(IA.LT.NOCTPA) THEN
            IA = IA + 1
          ELSE
            NONEW = 1
          END IF
        END IF
      END IF
*.Next block
      IATP = IA
      IBTP = IB
      IASM = ISM
C?    WRITE(6,*) ' IATP, IBTP, IASM = ', IATP, IBTP, IASM
      IBSM = ISMOST(IASM)
*. Length
      NSTA = NSSOA(IASM,IA)
      NSTB = NSSOB(IBSM,IB)
      LBLOCK= NSTA*NSTB
      IF(IDC.EQ.1.OR.IA.GT.IB.OR.(IA.EQ.IB.AND.IASM.GT.IBSM)) THEN
        LBLOCKP = NSTA*NSTB
      ELSE IF(IDC.EQ.2.AND.IA.EQ.IB.AND.IASM.EQ.IBSM) THEN
        LBLOCKP = NSTA*(NSTA+1)/2
      END IF
*
C?    WRITE(6,*) ' IASM IBSM IA IB LBLOCKP,LBLOCK' ,    
C?   &             IASM,IBSM,IA,IB,LBLOCKP,LBLOCK
*
      IF(NONEW.EQ.1) GOTO 1001
*. Should this block be included
      IF(IDC.EQ.2.AND.IA.LT.IB) GOTO 1000
      IF(IDC.EQ.2.AND.IA.EQ.IB.AND.IASM.LT.IBSM) GOTO 1000
      IF(IOCOC(IA,IB).EQ.0) GOTO 1000
 1001 CONTINUE
*
*
      NTEST = 00
      IF(NTEST.NE.0) THEN
        WRITE(6,'(A,4I4)')
     &  ' NXTBLK: ISM IA IB NONEW ', IASM,IA,IB,NONEW
      END IF
*
      RETURN
      END
* Output
      SUBROUTINE NXTIJ(I,J,NI,NJ,IJSM,NONEW)
*
* An ordered pair (I,J) is given ,I.LE.NI,J.LE.NJ
*
* Find next pair, if IJSM .ne. 0 ,I .ge. J
*
      NONEW = 0
  100 CONTINUE
      IF(I.LT.NI) THEN
        I = I + 1
      ELSE
        IF(J.LT.NJ) THEN
          I = 1
          J = J+1
        ELSE
          NONEW = 1
          GOTO 101
        END IF
      END IF
      IF(IJSM.NE.0.AND.I.LT.J) GOTO 100
  101 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' next (i,j) pair ', I,J
      END IF
*
      RETURN
      END
      SUBROUTINE NXTORD(INUM,NELMNT,MINVAL,MAXVAL,NONEW)
*
* An ordered set of numbers INUM(I),I=1,NELMNT is
* given in strictly ascending order. Values of INUM(*) is
* restricted to the interval MINVAL,MAXVAL .
*
* Find next higher number.
*
* NONEW = 1 on return indicates that no additional numbers
* could be obtained.
*
* Jeppe Olsen May 1989, special handling for NELMNT = 0, added March 2013..
*
      DIMENSION INUM(*)
*
       NTEST = 000
       IF( NTEST .NE. 0 ) THEN
         WRITE(6,*) ' Initial number in NXTORD '
         CALL IWRTMA(INUM,1,NELMNT,1,NELMNT)
       END IF
*
      IF(NELMNT.EQ.0) THEN
        NONEW = 1
        GOTO 2000
      END IF
*
      IPLACE = 0
 1000 CONTINUE
        IPLACE = IPLACE + 1
        IF( IPLACE .LT. NELMNT .AND.
     &      INUM(IPLACE)+1 .LT. INUM(IPLACE+1)
     &  .OR.IPLACE.EQ. NELMNT .AND.
     &      INUM(IPLACE)+1.LE.MAXVAL) THEN
              INUM(IPLACE) = INUM(IPLACE) + 1
              NONEW = 0
              GOTO 1001
        ELSE IF ( IPLACE.LT.NELMNT) THEN
              IF(IPLACE .EQ. 1 ) THEN
                INUM(IPLACE) = MINVAL
              ELSE
                INUM(IPLACE) = INUM(IPLACE-1) + 1
              END IF
        ELSE IF ( IPLACE. EQ. NELMNT ) THEN
              NONEW = 1
              GOTO 1001
        END IF
      GOTO 1000
 1001 CONTINUE
 2000 CONTINUE
*
      IF( NTEST .NE. 0 ) THEN
        IF(NONEW.EQ.0) THEN
          WRITE(6,*) ' New number '
          CALL IWRTMA(INUM,1,NELMNT,1,NELMNT)
        ELSE
          WRITE(6,*) ' No new number '
        END IF
      END IF
*
      RETURN
      END
      SUBROUTINE ORBINF(LUOUT,IPRNT)
*
* Obtain information about orbitals from shell information
*
* =====
* input
* =====
* Shell and symmetry information in /LUCINP/
*
* ======
* Output
* ======
* Orbital information in /ORBINP/
*
* Jeppe Olsen, Winter of 1991
*
COLD  INTEGER CITYP
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cgas.inc'
*
      INCLUDE 'orbinp.inc'
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
************************************************
*                                              *
* Part 1: From shell format to orbital format *
*                                              *
************************************************
      CALL OSPIR(NOSPIR,IOSPIR,PNTGRP,NIRREP,MXPIRR,MXPOBS,IPRNT)
*
* 2: Shell information to orbital information for each group of orbital
*
*
* ===============
*     GAS case
* ===============
*
        DO IGAS = 1, NGAS
*. Shell => orbitals for each GAS space
          CALL SHTOOB(NGSSH(1,IGAS),NIRREP,MXPOBS,NSMOB,NOSPIR,
     &                IOSPIR,NGSOB(1,IGAS),NGSOBT(IGAS))
        END DO
*. Inactive orbitals
        CALL SHTOOB(NINASH,NIRREP,MXPOBS,NSMOB,NOSPIR,
     &                IOSPIR,NINOBS(1),NINOB)
*. Secondary orbitals
        CALL SHTOOB(NSECSH,NIRREP,MXPOBS,NSMOB,NOSPIR,
     &                IOSPIR,NSCOBS(1),NSCOB)
*
*  ========================================================
*. Number of inactive, active, occupied , deleted orbitals
*  ========================================================
* 
*
*
       NTOOB = 0
       NACOB = 0
       NOCOB = 0
       IZERO = 0
       CALL ISETVC(NACOBS,IZERO,NSMOB)
       DO IGAS = 1, NGAS
         CALL IVCSUM(NACOBS,NACOBS,NGSOB(1,IGAS),1,1,NSMOB)
         NACOB = NACOB + NGSOBT(IGAS)
       END DO
       CALL IVCSUM(NOCOBS,NACOBS,NINOBS,1,1,NSMOB)
       NOCOB = NACOB + NINOB
       CALL IVCSUM(NTOOBS,NOCOBS,NSCOBS,1,1,NSMOB)
       NTOOB = NOCOB  + NSCOB
    
       
* ===============================================
*. Well, report back
* ===============================================
        IF(NTEST.GE.5) THEN
          WRITE(LUOUT,*)
          WRITE(LUOUT,*) ' Number of orbitals per symmetry:'
          WRITE(LUOUT,*) ' ================================='
          WRITE(LUOUT,*)
          WRITE(LUOUT,'(1H ,A,10I4,A)')
     &    '            Symmetry  ',(I,I = 1,NSMOB) 
          WRITE(LUOUT,'(1H ,A,2X,10A)')
     &    '           ========== ',('====',I = 1,NSMOB)
          WRITE(LUOUT,'(1H ,A,9X,10I4,8X,I3)')
     &      '   Inactive  ',(NINOBS(I),I=1,NSMOB),NINOB
          DO IGAS = 1, NGAS
            WRITE(LUOUT,'(1H      ,A,I3,7X,A,10I4,8X,I3)')
     &      '   GAS',IGAS,'      ',(NGSOB(I,IGAS),I=1,NSMOB),
     &      NGSOBT(IGAS)
          END DO
          WRITE(LUOUT,'(1H A,9X,10I4,8X,I3)')
     &      '   Secondary ',(NSCOBS(I),I=1,NSMOB),NSCOB
*
          WRITE(LUOUT,*) ' Total number of orbitals ', NTOOB
          WRITE(LUOUT,*) ' Total number of occupied orbitals ', NOCOB
        END IF
*. Offsets for orbitals of given symmetry
        ITOOBS(1) = 1
        DO  ISMOB = 2, NSMOB
          ITOOBS(ISMOB) = ITOOBS(ISMOB-1)+NTOOBS(ISMOB-1)
        END DO
*
        IF(NTEST.GE.5) THEN
          WRITE(6,*) ' Offsets for orbital of given symmetry '
          CALL IWRTMA(ITOOBS,1,NSMOB,1,NSMOB)
        END IF

********************************************
*                                          *
* Part 2: Reordering arrays for orbitals  *
*                                          *
********************************************
        CALL ORBORD_GAS(NSMOB,MXPOBS,MXPNGAS,NGAS,NGSOB,NGSOBT,
     &       NINOBS,NINOB,NSCOBS,NSCOB,
     &       NOCOBS,NTOOBS,NTOOB,
     &       IREOST,IREOTS,ISMFTO,ITPFSO,
     &       IBSO,NTSOB,IBTSOB,ITSOB,NOBPTS,IOBPTS,IOBPTS_AC,
     &       ISMFSO,ITPFTO,NOBPT,NOBPTS_GN, IOBPTS_GN, IPRNT)
*. Largest number of orbitals of given sym and type
      MXTSOB = 0
      MXTOB = 0
      DO IOBTP = 0, NGAS+1
        LTOB = 0
        DO IOBSM = 1, NSMOB
         MXTSOB = MAX(MXTSOB,NOBPTS_GN(IOBTP,IOBSM))
         LTOB = LTOB + NOBPTS_GN(IOBTP,IOBSM)
        END DO
        MXTOB= MAX(LTOB,MXTOB)
      END DO
C?    WRITE(6,*) ' MXTSOB,MXTOB from ORBINF = ', MXTSOB,MXTOB
*
      RETURN
      END
      SUBROUTINE ORBORD_GAS(NSMOB,MXPOBS,MXPNGAS,NGAS,NGSOB,NGSOBT,
     &                  NINOBS,NINOB,NSCOBS,NSCOB,
     &                  NOCOBS,NTOOBS,NTOOB,
     &                  IREOST,IREOTS,ISFTO,ITFSO,IBSO,
     &                  NTSOB,IBTSOB,ITSOB,NOBPTS,IOBPTS,IOBPTS_AC,
     &                  ISFSO,ITFTO,NOBPT,NOBPTS_GN,IOBPTS_GN,IPRNT)
*
* Obtain Reordering arrays for orbitals
* ( See note below for assumed ordering )
*
*
* GAS version 
*
* =====
* Input
* =====
*  NSMOB  : Number of orbital symmetries
*  MXPOBS : Max number of orbital symmetries allowed by program
*  MXPNGAS: Max number of GAS spaces allowed by program
*  NGAS   : Number of GAS spaces
*  NGSOB  : Number of GAS orbitals per symmetry and space
*  NGSOBT : Number of GAS orbitals per space
*  NOCOBS : Number of occupied orbitals per symmetry
*  NTOOBS : Number of orbitals per symmetry,all types
*
* ======
* Output
* ======
*  IREOST: Reordering array symmetry => type
*  IREOTS: Reordering array type     => symmetry
*  ISFTO  : Symmetry array for type ordered orbitals
*  ITFSO  : Type array for symmetry ordered orbitals( not activated )
*  IBSO   : First orbital of given symmetry ( symmetry ordered )
*  NOBPTS : Number of orbitals per subtype and symmetry
*  IOBPTS : Off sets for orbitals of given subtype and symmetry
*           ordered according to input integrals
*  NOBPTS_GN: As NOBPTS, but includes info for inactive and secondary
*  IOBPTS_GN: As IOBPTS, but includes info for inactive and secondary
*  IOBPTS_AC: Offsets for orbitals of given subtype and symmetry,
*             when only active orbitals are considered
*             
*
* ISFSO  : Symmetry of orbitals, symmetry ordereing
* ITFTO  : Type of orbital, type ordering
*
* Jeppe Olsen, Winter 1994
*              Explicit introduction of inactive and secondary orbitals:
*              June 2010...
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION NGSOB(MXPOBS,MXPNGAS),NOCOBS(*),NTOOBS(*)
      DIMENSION NGSOBT(MXPNGAS)
      DIMENSION NINOBS(MXPOBS),NSCOBS(MXPOBS)
*. Output
      DIMENSION IREOST(*),IREOTS(*),ISFTO(*),ITFSO(*),IBSO(*)
      DIMENSION ISFSO(*),ITFTO(*)
      DIMENSION NOBPTS(MXPNGAS ,*),IOBPTS(MXPNGAS ,*)
      DIMENSION NOBPTS_GN(0:MXPNGAS,*),IOBPTS_GN(0:MXPNGAS,*)
      INTEGER IOBPTS_AC(MXPNGAS,*)
      DIMENSION NOBPT(MXPNGAS )
 
* ==========================
* Note on order of orbitals
* ==========================
*
* The orbitals are supposed to be imported ordered symmetry-type
* ordered as
*
* Loop over symmetries of orbitals
*  Inactive orbitals of this symmetry
*  Loop over GAS spaces  
*   Loop over orbitals of this sym and GAS
*   End of Loop over orbitals
*  End of Loop over Gas spaces
*  Secondary orbitals of this symmetry
* End of loop over symmetries
*
* Internally the orbitals are reordered to type symmetry order
* where the outer loop is over types and the inner loop is
* over symmetries, i.e.
*
*.Loop over symmetries of orbitals
*  Loop over inactive orbitals of this symmetry
*. End of loop over orbitals
* End of loop over symmetries
* Loop over GAS spaces  
*  Loop over symmetries of orbitals
*   Loop over orbitals of this sym and GAS
*   End of Loop over orbitals
*  End of loop over symmetries
* End of Loop over Gas spaces
*.Loop over symmetries of orbitals
*  Loop over secondary orbitals of this symmetry
*  End of loop over secondary orbitals of given symmetry
* End of loop over symmetries
*
*. 1:  Construct ISFTO, ITFTO, IREOST,IREOTS,NOBPTS,IOBPTS, IOBPTS_AC
*  Note: IOBPTS is absolute and includes gaps in orbitalnumbering
*        arising from inactive and secondary orbitals
*        IOBPTS_AC is when only the active orbitals are considered
*        and has no gaps between active orbitals
*
* Inactive orbitals have type 0, secondary orbitals type NGAS + 1
      ITSOFF = 1
      DO IGAS = 0, NGAS + 1
        DO ISYM = 1, NSMOB
          IF(ISYM.EQ.1) THEN
            IBSSM = 1
          ELSE
            IBSSM = IBSSM + NTOOBS(ISYM-1)
          END IF
          IF(IGAS.EQ.0) THEN
            NPREV = 0
          ELSE
            NPREV = NINOBS(ISYM)
            DO JGAS = 1, IGAS-1
              NPREV = NPREV + NGSOB(ISYM,JGAS)
            END DO
          END IF
          IADD = 0
          IF(IGAS.NE.0.AND.IGAS.NE.NGAS+1) THEN
            NOBPTS(IGAS,ISYM) = NGSOB(ISYM,IGAS)
            IOBPTS(IGAS,ISYM) = ITSOFF
          END IF
*
          IF(IGAS.EQ.0) THEN
            NORB_L = NINOBS(ISYM)
          ELSE IF(IGAS.LE.NGAS) THEN
            NORB_L = NGSOB(ISYM,IGAS)
          ELSE
            NORB_L = NSCOBS(ISYM)
          END IF
          NOBPTS_GN(IGAS,ISYM) = NORB_L
          IOBPTS_GN(IGAS,ISYM) = ITSOFF
*
          DO IORB = ITSOFF, ITSOFF+NORB_L-1
            IADD = IADD + 1
C?          WRITE(6,*) ' IORB, IADD, IBSSM, NPREV, IADD =',
C?   &                   IORB, IADD, IBSSM, NPREV, IADD
            IREOTS(IORB) = IBSSM-1+NPREV+IADD
            IREOST(IBSSM-1+NPREV+IADD) = IORB
            ITFTO(IORB) = IGAS
            ISFTO(IORB) = ISYM
          END DO
          ITSOFF = ITSOFF + NORB_L
        END DO
      END DO
*
*. IOBPTS_AC
*
      ITSOFF = 1
      DO IGAS = 1, NGAS 
        DO ISYM = 1, NSMOB
          IOBPTS_AC(IGAS,ISYM) = ITSOFF
          NORB_L = NGSOB(ISYM,IGAS)
          ITSOFF = ITSOFF + NORB_L
        END DO
      END DO
*
* 2 : ISFSO,ITFSO
*
      ISTOFF = 1
      DO ISYM = 1, NSMOB
        DO IGAS = 0, NGAS+1
          IF(IGAS.EQ.0) THEN
            NORB_L = NINOBS(ISYM)
          ELSE IF(IGAS.LE.NGAS) THEN
            NORB_L = NGSOB(ISYM,IGAS)
          ELSE
            NORB_L = NSCOBS(ISYM)
          END IF
          DO IORB = ISTOFF,ISTOFF+NORB_L-1
            ISFSO(IORB) = ISYM
            ITFSO(IORB) = IGAS
          END DO
          ISTOFF = ISTOFF + NORB_L
        END DO
      END DO
*
* 3 IBSO, NOBPT
*
      IOFF = 1
      DO ISM = 1, NSMOB
       IBSO(ISM) = IOFF
       IOFF = IOFF + NTOOBS(ISM)
      END DO
      DO IGAS = 1, NGAS
        NOBPT(IGAS) = NGSOBT(IGAS)
      END DO
*
      NTEST = 00
      NTEST = MAX(IPRNT,NTEST)
      IF( NTEST .GE. 5 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' ==================='
        WRITE(6,*) ' Output from ORBORD '
        WRITE(6,*) ' ==================='
        WRITE(6,*)
        WRITE(6,*) ' Symmetry of orbitals , type ordered '
        CALL IWRTMA(ISFTO,1,NTOOB,1,NTOOB)
        WRITE(6,*) ' Symmetry => type reordering array '
        CALL IWRTMA(IREOST,1,NTOOB,1,NTOOB)
        WRITE(6,*) ' Type => symmetry reordering array '
        CALL IWRTMA(IREOTS,1,NTOOB,1,NTOOB)
        WRITE(6,*) ' IBSO array '
        CALL IWRTMA(IBSO,1,NSMOB,1,NSMOB)
*
        WRITE(6,*) ' NOBPTS '
        CALL IWRTMA(NOBPTS,NGAS,NSMOB,MXPNGAS,MXPOBS)
        WRITE(6,*) ' NOBPT '
        CALL IWRTMA(NOBPT,NGAS,1,MXPNGAS,1)
        WRITE(6,*) ' IOBPTS '
        CALL IWRTMA(IOBPTS,NGAS,NSMOB,MXPNGAS,MXPOBS)
        WRITE(6,*) ' IOBPTS_AC '
        CALL IWRTMA(IOBPTS_AC,NGAS,NSMOB,MXPNGAS,MXPOBS)
        WRITE(6,*) ' IOBPTS_GN '
        CALL IWRTMA(IOBPTS_GN,NGAS+2,NSMOB,MXPNGAS+1,MXPOBS)
        WRITE(6,*) ' NOBPTS_GN '
        CALL IWRTMA(NOBPTS_GN,NGAS+2,NSMOB,MXPNGAS+1,MXPOBS)
*
        WRITE(6,*) ' ISFTO array: '
        CALL IWRTMA(ISFTO,1,NTOOB,1,NTOOB)
        WRITE(6,*) ' ITFSO array: '
        CALL IWRTMA(ITFSO,1,NTOOB,1,NTOOB)
*
        WRITE(6,*) ' ISFSO array: '
        CALL IWRTMA(ISFSO,1,NTOOB,1,NTOOB)
        WRITE(6,*) ' ITFTO array: '
        CALL IWRTMA(ITFTO,1,NTOOB,1,NTOOB)
      END IF
*
       
      RETURN
      END
      SUBROUTINE ORDSTR(IINST,IOUTST,NELMNT,ISIGN,IPRNT)
C
C ORDER A STRING OF INTEGERS TO ASCENDING ORDER
C
C IINST: INPUT STRING IS IINST
C IOUTST: OUTPUT STRING IS IOUTST
C NELMNT: NUMBER OF INTEGERS IN STRING
C ISIGN:  SIGN OF PERMUTATION: + 1: EVEN PERMUTATIONN
C                                - 1: ODD  PERMUTATION
C
C THIS CODE CONTAINS THE OLD ORDER CODE OF JOE GOLAB
C ( HE IS HEREBY AKNOWLEDGED , AND I AM EXCUSED )
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION IINST(NELMNT),IOUTST(NELMNT)
C
      NTEST = 00
      IF(NTEST.GE.100) THEN
       WRITE(6,*) ' Number of elements to be sorted ', NELMNT
       WRITE(6,*) ' And the elements ' 
       CALL IWRTMA(IINST,1,NELMNT,1,NELMNT)
      END IF
*
      ISIGN = 1
      IF(NELMNT.EQ.0) GOTO 50
*
      CALL ICOPVE(IINST,IOUTST,NELMNT)
      ISIGN = 1
C
C       BEGIN TO ORDER
C
        JOE = 1
  10    I = JOE
  20    CONTINUE
        IF(I.EQ.NELMNT) GO TO 50
        IF(IOUTST(I).LE.IOUTST(I+1)) GO TO 40
        JOE = I + 1
  30    SWAP = IOUTST(I)
        ISIGN = - ISIGN
        IOUTST(I) = IOUTST(I+1)
        IOUTST(I+1) = SWAP
        IF(I.EQ.1) GO TO 10
        I = I - 1
        IF(IOUTST(I).GT.IOUTST(I+1)) GO TO 30
        GO TO 10
 40     I = I + 1
      GO TO 20
C
C     END ORDER
C
 50   CONTINUE
      NTEST = 000
      NTEST = MAX(NTEST,IPRNT)
      IF( NTEST .GE.200) THEN
        WRITE(6,*)  ' INPUT STRING ORDERED STRING ISIGN '
        CALL IWRTMA(IINST,1,NELMNT,1,NELMNT)
        CALL IWRTMA(IOUTST,1,NELMNT,1,NELMNT)
        WRITE(6,*) ' ISIGN: ', ISIGN
      END IF
C
      RETURN
      END
      SUBROUTINE OSPIR(NOSPIR,IOSPIR,PNTGRP,NIRREP,MXPIRR,MXPOBS,IPRNT)
*
* Number and symmetries of orbitals corresponding to a given shell
*
* =====
* Input
* =====
*
*   PNTGRP : type of pointgroup
*         = 1 => D2h or a subgroup of D2H
*         = 2 => C inf v
*         = 3 => D inf h
*         = 4 => O 3
*   NIRREP: Number of irreducible representations per point group
*   MXPIRR: Largest allowed number of shell irreps
*   MXPOBS: Largest allowed number of orbital symmetries
*
* ======
* Output
* ======
*
*   NOSPIR: Number of orbital symmetries per irrep
*   IOSPIR: Orbital symmetries corresponding to a given irrep
*
* Jeppe Olsen , Winter of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER PNTGRP
*. Output
      DIMENSION NOSPIR(MXPIRR),IOSPIR(MXPOBS,MXPIRR)
*
      IF(PNTGRP.EQ.1) THEN
*=====
*.D2h
*=====
        NSMOB = 0
        DO 10 IRREP = 1, 8
          NOSPIR(IRREP) = 1
          IOSPIR(1,IRREP) = IRREP
   10   CONTINUE
      ELSE IF(PNTGRP.EQ.2) THEN
* =========
*. C inf V
* =========
* orbital symmetry is numbered as IML - MNMLOB + 1
        MNMLOB = -(NIRREP-1)
        DO 20 IRREP = 1, NIRREP
*.Irrep I contains orbitals with ML = -(IRREP-1),+(IRREP-1)
          IF(IRREP.EQ.1) THEN
            NOSPIR(IRREP) = 1
            IOSPIR(1,IRREP) = IRREP - 1 - MNMLOB + 1
          ELSE
            NOSPIR(IRREP) = 2
            IOSPIR(1,IRREP) = -(IRREP - 1) - MNMLOB + 1
            IOSPIR(2,IRREP) =  (IRREP - 1) - MNMLOB + 1
          END IF
   20   CONTINUE
      ELSE IF(PNTGRP.EQ.3) THEN
* ========
*. D inf H
* ========
* orbital symmetry is numbered as (PARITY-1) * NMLOB + IML - MNMLOB + 1
        MXMLOB =  NIRREP/2-1
        MNMLOB = -MXMLOB
        NMLOB =   NIRREP - 1
        IRREP = 0
        DO 35 IPARI = 1, 2
          IADD = (IPARI-1)*NMLOB
          DO 30 ML = 0,MXMLOB
            IRREP = IRREP + 1
            IF(ML.EQ.0) THEN
              NOSPIR(IRREP) = 1
              IOSPIR(1,IRREP) = IADD + ML - MNMLOB + 1
            ELSE
              NOSPIR(IRREP) = 2
              IOSPIR(1,IRREP) = IADD - ML - MNMLOB + 1
              IOSPIR(2,IRREP) = IADD + ML - MNMLOB + 1
            END IF
   30     CONTINUE
   35   CONTINUE
 
      ELSE IF(PNTGRP.EQ.4) THEN
* =====
*. O 3
* =====
* orbital symmetry is numbered as (PARITY-1) * NMLOB + IML - MNMLOB + 1
        MXMLOB =  NIRREP/2-1
        MNMLOB = -MXMLOB
        NMLOB =   NIRREP - 1
        DO 45 L = 0, NIRREP - 1
          IF(MOD(L,2).EQ.0) THEN
            IPARI = 1
          ELSE
            IPARI = 2
          END IF
          IADD = (IPARI-1)*NMLOB
          IRREP = L + 1
          NOSPIR(IRREP) = 2 * L + 1
          ICOMP = 0
          DO 40 ML = MNMLOB,MXMLOB
            ICOMP = ICOMP + 1
            IOSPIR(ICOMP,IRREP) = IADD + ML - MNMLOB + 1
   40     CONTINUE
   45   CONTINUE
      ELSE
        WRITE(6,*) ' Sorry  PNTGRP out of range , PNTGRP = ', PNTGRP
        WRITE(6,*) ' ORBIRR fatally wounded '
        STOP 5
      END IF
*
      NTEST = 0
      NTEST = MAX(IPRNT,NTEST)
      IF(NTEST.GE.5) THEN
        WRITE(6,*) ' OSPIR speaking '
        WRITE(6,*) ' ================'
        WRITE(6,*)
        WRITE(6,*) ' NTEST = ', NTEST
        WRITE(6,*) ' Number of orbitals per irrep '
        CALL IWRTMA(NOSPIR,1,NIRREP,1,NIRREP)
        WRITE(6,*) ' Orbital symmetries per irrep '
        DO 100 IRREP = 1, NIRREP
          CALL IWRTMA(IOSPIR(1,IRREP),1,NOSPIR(IRREP),1,NOSPIR(IRREP))
  100   CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE PICO(VEC1,VEC2,VEC3,LU1,LU2,LU3,RNRM,EIG,FINEIG,
     &                MAXIT,LUDIA,NROOT,WORK,IPRT,LBLK,
     &                ISUB,NSUB,HSUB,NP1,NP2,NQ,CSUB,HCSUB,EIGSHF)
*
* Perturbative eigen value solver designed to
* minimize I/O
*
*
* LBLK defines structure of the files containig vectors
* This subroutine is independent of the choice of LBLK
*
* Micro Davidson algorihm  with 3 vector segements in core.
* Single root version
*
* ======
* Input:
* ======
*
*     LU1: Contains initial set of vectors
*     VEC1,VEC2 VEC3: SEGMENTS,EACH MUST BE ABLE TO HOLD
*                    LARGEST SEGMENT OF VECTOR
*     LU2    : Sigma vector file
*     LU3   : Scratch file
*     LUDIA: File containing CI diagonal
*     NROOT: Number of CI vectors to be obtained  =  1
*
* Subspace
*     ISUB: elements of subspace 
*     NSUB: Number of elements in subspace
*     HSUB: subspace matrix
*     NP1,NP2,NQ: Size of the three dimensions of subspace
*
      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 INPRDD
      DIMENSION RNRM(MAXIT,1),EIG(MAXIT,1)
      LOGICAL CONVER
*. Input
      DIMENSION ISUB(*),HSUB(*)
*. Scratch 
      DIMENSION CSUB(*),HCSUB(*)
      DIMENSION VEC1(*),VEC2(*),VEC3(*)
      DIMENSION WORK(*)
*
C?    WRITE(6,*) ' IPRT = ',IPRT
      WRITE(6,*) ' Diagonalizer PICO at your service '
      WRITE(6,*) ' LU1 LU2 LU3 ',LU1,LU2,LU3
      IF( NROOT.GT.1) THEN
        WRITE(6,*) ' Sorry PICO wounded , NROOT  .GT. 1 '
        STOP ' Enforced stop in PICO'
      END IF
*
      SCALEP = 1.0D0
      TEST = 1.0D-5
      THRES= 1.0D-5
      CONVER = .FALSE.
C?    WRITE(6,*) ' LBLK from PICO ', LBLK
*
      ITER = 0
 1000 CONTINUE
        ITER = ITER + 1
        IF(IPRT.GE.5)
     & WRITE(6,*) ' Info from iteration .... ', ITER
*
*===========================
*. MATRIX TIMES TRIAL VECTOR
*===========================
*
       CALL MV7(VEC1,VEC2,LU1,LU2,0,0)
        IF ( IPRT  .GE. 101) THEN
          WRITE(6,*) '  C     VECTOR ACCORDING TO PICO '
          CALL WRTVCD(VEC1,LU1,1,LBLK)
          WRITE(6,*) '  SIGMA VECTOR ACCORDING TO PICO '
          CALL WRTVCD(VEC1,LU2,1,LBLK)
        END IF
*
*==========================
*. ENERGY AND RESIDUAL NORM
*==========================
*
        CALL ERES(LU1,LU2,VEC1,VEC2,LBLK,CC,CHC,CHHC,
     &           NSUB,ISUB,CSUB,HCSUB,IPRT)
 
        EIGAPR = CHC/CC
        ZERO = 0.0D0
        RESNRM = SQRT(MAX(CHHC-EIGAPR ** 2 * CC,ZERO))
        IF(IPRT.GE.3) WRITE(6,'(A,2E15.7)')
     &  ' Current energy and residual ', EIGAPR+EIGSHF,RESNRM
        EIG(ITER,1) = EIGAPR
        RNRM(ITER,1) = RESNRM
*. FIND OPTIMAL CHOICE OF STEP IN PREVIOUS ITERATION
        IF(RESNRM.GT.THRES.AND.ITER.LT.MAXIT) THEN
*
*==========================
*. NEW SOLUTION VECTOR
*==========================
*. Obtain (H0-E)**(-1)/
*. GAMMA, DELTA AND EPSIL FOR CORRECTION
          CALL GDEPS(LU1,LU2 ,LUDIA,ISUB,ESUB,VSUB,NSUB,WORK,
     &               EIGAPR,GAMMA,DELTA,EPSIL,VEC1,VEC2,VEC3,
     *               HCSUB,CSUB,LBLK,IPRT)
*. SHOULD INVERSE ITERATION MODIFICATION BE APPLIED
          IF(ABS(EPSIL).GE.1.0D-9) THEN
            CORREC = GAMMA/DELTA
          ELSE
            CORREC = 0.0D0
          END IF
         IF(IPRT.GE.5) WRITE(6,*) ' CORREC = ', CORREC
*.  FORM NEW SOLUTION APPROXIMATION AS
*.  SCALE*(C - (H0-E)*(HC -(EIGAPR+CORREC)C) WHERE SCALE NORMALIZES
*.  PREVIOUS VECTOR
*  .C-(H0-E)**-1(HC-(EIGAPR+CORREC)CIN SUBSPACE
          KLVEC1 = 1
          KLVEC2 = KLVEC1 + NSUB
          KLVEC3 = KLVEC2 + NSUB
*. HC -(E+CORREC)C IN KLVEC1
          CALL VECSUM(WORK(KLVEC1),HCSUB,CSUB,1.0D0,-(EIGAPR+CORREC),
     &               NSUB)
          CALL XDXTV(WORK(KLVEC2),WORK(KLVEC1),VSUB,ESUB,NSUB,
     &              WORK(KLVEC3),-EIGAPR,1)
          CALL VECSUM(WORK(KLVEC1),CSUB,WORK(KLVEC2),1.0D0,-1.0D0,NSUB)
*. IN FULL SPACE , SAVE ON LU3
          SCALE = 1.0D0/SQRT(CC)
          CALL RESID(LU1,LU2,LU3,LUDIA,LBLK,
     &    VEC1,VEC2,VEC3,NSUB,ISUB,WORK(KLVEC1),SCALE,
     &    EIGAPR,EIGAPR+CORREC)
* CHECK OVERLAP BETWEEN NEW AND OLD VECTORS
          XNORM = INPRDD(VEC1,VEC2,LU1,LU3,1,LBLK)
          IF(IPRT.GE.5)
     &    WRITE(6,*) ' OVERLAP BETWEEN OLD AND NEW VECTOR ',XNORM/SCALE
          CALL COPVCD(LU3,LU1,VEC1,1,LBLK)
          GOTO 1000
       ELSE
*================================
*PROCESSING OF FINAL WAVEFUNCTION
*================================
          IF(RESNRM.LE.THRES) THEN
            CONVER = .TRUE.
          ELSE
            CONVER = .FALSE.
          END IF
          XNORM = INPRDD(VEC1,VEC1,LU1,LU1,1,LBLK)
          SCALE = 1.0D0/SQRT(XNORM )
          CALL SCLVCD(LU1,LU3,SCALE,VEC1,1,LBLK)
          CALL COPVCD(LU3,LU1,VEC1,1,LBLK)
       END IF
*. END OF LOOP OVER ITERATIONS
*
      IF( .NOT. CONVER ) THEN
*.       CONVERGENCE WAS NOT OBTAINED
         IF(IPRT .NE. 0 )
     &   WRITE(6,1170) MAXIT
 1170    FORMAT('0  Convergence was not obtained in ',I3,' iterations')
      ELSE
*.       CONVERGENCE WAS OBTAINED
         IF (IPRT .NE. 0 )
     &   WRITE(6,1180) ITER
 1180    FORMAT(1H0,' Convergence was obtained in ',I3,' iterations')
        END IF
*
      IF ( IPRT .GE. 0 ) THEN
        CALL REWINE(LU1,LBLK)
          WRITE(6,*)
          WRITE(6,'(A,I3)')
     &  ' Information about convergence for root... ' ,1
          WRITE(6,*)
     &    '============================================'
          WRITE(6,*)
          FINEIG = EIG(ITER,1)
          WRITE(6,1190) FINEIG+EIGSHF
 1190     FORMAT(' The final approximation to eigenvalue ',F18.10)
          WRITE(6,1300)
 1300     FORMAT(1H0,' Summary of iterations ',/,1H
     +          ,' ----------------------')
          WRITE(6,1310)
 1310     FORMAT
     &    (1H0,' Iteration point      Eigenvalue         Residual ')
          DO 1330 I=1,ITER
 1330     WRITE(6,1340) I,EIG(I,1)+EIGSHF,RNRM(I,1)
 1340     FORMAT(1H ,6X,I4,8X,F18.13,2X,E12.5)
*
          IF(IPRT .GE. 100) THEN
            WRITE(6,'(A)')
     &      '  Final approximation to eigenvector '
            CALL WRTVCD(VEC1,LU1,0,LBLK)
          END IF
      ELSE
           FINEIG = EIG(ITER,1)
      END IF
C
      RETURN
C1030 FORMAT(1H0,2X,7F15.8,/,(1H ,2X,7F15.8))
C1120 FORMAT(1H0,2X,I3,7F15.8,/,(1H ,5X,7F15.8))
      END
      SUBROUTINE PMPLFM(AP,B,NDIM)
*
* Add lower half of a full matrix to a matrix packed
* in lower triangular form ( packed matrix stored columnwise )
*
      IMPLICIT REAL*8           ( A-H,O-Z)
      DIMENSION AP(*),B(*)
*
      IBSP = 1
      IBSF = 1
      DO 100 ICOL = 1, NDIM
        NELMNT = NDIM - ICOL + 1
        CALL VECSUM(AP(IBSP),AP(IBSP),B(IBSF),1.0D0,1.0D0,NELMNT)
        IBSP = IBSP + NELMNT
        IBSF = IBSF + NDIM
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE PNT2DM(I12SM,NSMOB,NSMSX,OSXO,IPSM,JPSM,
     &                  IJSM,ISM2,IPNTR,MXPOBS)
*
* Pointer to two dimensional array
*
* =====
* Input
* =====
* I12SM : ne.0 => restrict to lower half
*          eq.0 => complete matrix
* NSMOB: Number of orbital symmetries
* NSMSX: Number of SX      symmetries
* OSXO : Symmetry of orbital, SX => symmetry of other orbital
* IPSM: Number of orbitals per symmetry for index 1
* JPSM: Number of orbitals per symmetry for index 2
* IJSM : Symmetry of two index array
*
* =======
* Output
* =======
* IPNTR: Pointer to block with first index of given symmetry
*         = 0 indicates forbidden block
* ISM2  symmetry of second index for given first index
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      INTEGER OSXO(MXPOBS,2*MXPOBS),IPSM(*),JPSM(*)
*.Output
      DIMENSION IPNTR(*),ISM2(*)
*
      CALL ISETVC(IPNTR,0,NSMOB)
      CALL ISETVC(ISM2 ,0,NSMOB)
      IOFF = 1
      DO 100 ISM = 1,NSMOB
        JSM = OSXO(ISM,IJSM)
        IF(JSM.EQ.0) GOTO 100
        IF(I12SM.EQ.0.OR.ISM.GE.JSM) THEN
*. Allowed block
          IPNTR(ISM) = IOFF
          ISM2(ISM) = JSM
          IF(I12SM.GT.0.AND.ISM.EQ.JSM) THEN
            IOFF = IOFF + IPSM(ISM)*(IPSM(ISM)+1)/2
          ELSE
            IOFF = IOFF + IPSM(ISM)*JPSM(JSM)
          END IF
        END IF
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.GE.1) THEN
        WRITE(6,*) ' dimension of two-dimensional array ',IOFF-1
      END IF
      IF(NTEST.GE.5) THEN
        WRITE(6,*) ' Pointer '
        CALL IWRTMA(IPNTR,1,NSMOB,1,NSMOB)
        WRITE(6,*) ' Symmetry of other array '
        CALL IWRTMA(ISM2,1,NSMOB,1,NSMOB)
      END IF
*
      RETURN
      END
      SUBROUTINE PNT4DM(NSMOB,NSMSX,MXPOBS,NO1PS,NO2PS,NO3PS,NO4PS,
     &           IDXSM,ADSXA,SXDXSX,IS12,IS34,IS1234,IPNTR,ISM4A,
     &           ADASX,NINT4D)
*
* Pointer for 4 dimensionl array with total symmetry IDXSM
* Pointer is given as 3 dimensional array corresponding
* to the first 3 indeces
* Symmetry of last index is give by ISM4
*
* IS12 (0,1,-1)   : Permutational symmetry between indeces 1 and 2
* IS34 (0,1,-1)   : Permutational symmetry between indeces 3 and 3
* IS1234 (0,1,-1) : permutational symmetry between indeces 12 and 34
*
*. General input
      INTEGER ADSXA(MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
      INTEGER ADASX(MXPOBS,MXPOBS)
*. Specific input
      INTEGER NO1PS(*),NO2PS(*),NO3PS(*),NO4PS(*)
*.Output
      INTEGER IPNTR(NSMOB,NSMOB,NSMOB),ISM4A(NSMOB,NSMOB,NSMOB)
*
      CALL ISETVC(IPNTR,0,NSMOB ** 3 )
      CALL ISETVC(ISM4A,0,NSMOB ** 3 )
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Info from PNT4DM '
        WRITE(6,*) ' ================='
        WRITE(6,*)
        WRITE(6,*) ' IS12, IS34, IS1234 =', IS12, IS34, IS1234
        WRITE(6,*) ' IDXSM = ', IDXSM
        WRITE(6,*) 'NO1PS NO2PS NO3PS NO4PS '
        CALL IWRTMA(NO1PS,1,NSMOB,1,NSMOB)
        CALL IWRTMA(NO2PS,1,NSMOB,1,NSMOB)
        CALL IWRTMA(NO3PS,1,NSMOB,1,NSMOB)
        CALL IWRTMA(NO4PS,1,NSMOB,1,NSMOB)
      END IF
*
      IOFF= 1
      N12 = 0
      N34 = 0
*
      DO 10 I1SM = 1, NSMOB
        DO 20 I2SM = 1, NSMOB
C?        WRITE(6,*) ' I1SM, I2SM = ', I1SM, I2SM
          I12SM = ADASX(I1SM,I2SM)
C?        WRITE(6,*) ' I12SM = ', I12SM
          I34SM = SXDXSX(I12SM,IDXSM)
C?        WRITE(6,*) ' I34SM = ', I34SM
          IF(I34SM.EQ.0) GOTO 20
          IF(IS12.NE.0.AND.I1SM.LT.I2SM) GOTO 20
          IF(IS12.EQ.0) THEN
           I12NUM = (I1SM-1)*NSMOB+I2SM
          ELSE
           I12NUM =  I1SM*(I1SM+1)/2+I2SM
          END IF
          IF(IS12.EQ.0.OR.I1SM.NE.I2SM) THEN
            N12 = NO1PS(I1SM)*NO2PS(I2SM)
          ELSE IF(IS12.EQ.1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
          ELSE IF(IS12.EQ.-1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
          END IF
          DO 30 I3SM = 1, NSMOB
            I4SM = ADSXA(I3SM,I34SM)
            IF(I4SM.EQ.0) GOTO 30
            IF(IS34.NE.0.AND.I3SM.LT.I4SM) GOTO 30
            IF(IS34.EQ.0) THEN
             I34NUM = (I3SM-1)*NSMOB+I4SM
            ELSE
             I34NUM =  I3SM*(I3SM+1)/2+I4SM
            END IF
            IF(IS1234.NE.0.AND.I12NUM.LT.I34NUM) GOTO 30
            IF(IS34.EQ.0.OR.I3SM.NE.I4SM) THEN
              N34 = NO3PS(I3SM)*NO4PS(I4SM)
            ELSE IF(IS34.EQ.1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
            ELSE IF(IS34.EQ.-1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
            END IF
            IF(IS1234.EQ.0.OR.I12NUM.NE.I34NUM) THEN
              IPNTR(I1SM,I2SM,I3SM) = IOFF
              ISM4A(I1SM,I2SM,I3SM) = I4SM
              IOFF= IOFF+ N12 * N34
            ELSE IF( IS1234.EQ.1.AND.I12NUM.EQ.I34NUM) THEN
              IPNTR(I1SM,I2SM,I3SM) = IOFF
              ISM4A(I1SM,I2SM,I3SM) = I4SM
              IOFF= IOFF + N12*(N12+1)/2
            ELSE IF( IS1234.EQ.-1.AND.I12NUM.EQ.I34NUM) THEN
              IPNTR(I1SM,I2SM,I3SM) = IOFF
              ISM4A(I1SM,I2SM,I3SM) = I4SM
              IOFF=  IOFF+ N12*(N12-1)/2
            END IF
C?          WRITE(6,*) ' I1SM I2SM I3SM I4SM    IOFF'
C?          WRITE(6,'(1H ,4I4,I9)')   I1SM,I2SM,I3SM,I4SM,IOFF
   30       CONTINUE
   20     CONTINUE
   10   CONTINUE
*
      NINT4D = IOFF - 1
*
*
C?    WRITE(6,*) ' PNT4DM , 64 elemets of IPNTR '
C?    call IWRTMA(IPNTR,1,64,1,64)
      NTEST = 0
      IF(NTEST.NE.0) THEN
         WRITE(6,*) ' Length of 4 index array ', NINT4D
      END IF
*
      RETURN
      END
      SUBROUTINE PNT4DM2(NEL,IMODE,
     &           NSMOB,NSMSX,MXPOBS,NO1PS,NO2PS,NO3PS,NO4PS,
     &           IDXSM,ADSXA,SXDXSX,IS12,IS34,IS1234,IPNTR,ISM4A,
     &           ADASX)
*
* Pointer for 4 dimensionl array with total symmetry IDXSM
* Pointer is given as 3 dimensional array corresponding
* to the first 3 indeces
* Symmetry of last index is give by ISM4
* on NEL the length of the 4-index array is returned
* if IMODE.EQ.0, only NEL is returned (IPNTR and ISM4A may be dummies then)
*
* IS12 (0,1,-1)   : Permutational symmetry between indeces 1 and 2
* IS34 (0,1,-1)   : Permutational symmetry between indeces 3 and 3
* IS1234 (0,1,-1) : permutational symmetry between indeces 12 and 34
*
*. General input
      INTEGER ADSXA(MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
      INTEGER ADASX(MXPOBS,MXPOBS)
*. Specific input
      INTEGER NO1PS(*),NO2PS(*),NO3PS(*),NO4PS(*)
*.Output
      INTEGER IPNTR(NSMOB,NSMOB,NSMOB),ISM4A(NSMOB,NSMOB,NSMOB)
*
      IF (IMODE.NE.0) THEN
        CALL ISETVC(IPNTR,0,NSMOB ** 3 )
        CALL ISETVC(ISM4A,0,NSMOB ** 3 )
      END IF
*
C?    WRITE(6,*) 'NO1PS NO2PS NO3PS NO4PS '
C?    CALL IWRTMA(NO1PS,1,NSMOB,1,NSMOB)
C?    CALL IWRTMA(NO2PS,1,NSMOB,1,NSMOB)
C?    CALL IWRTMA(NO3PS,1,NSMOB,1,NSMOB)
C?    CALL IWRTMA(NO4PS,1,NSMOB,1,NSMOB)
      IOFF= 1
      N12 = 0
      N34 = 0
      MM = 3006
*
      DO I1SM = 1, NSMOB
        DO I2SM = 1, NSMOB
          I12SM = ADASX(I1SM,I2SM)
          I34SM = SXDXSX(I12SM,IDXSM)
          IF(I34SM.EQ.0) CYCLE
          IF(IS12.NE.0.AND.I1SM.GT.I2SM) CYCLE
c          IF(IS12.EQ.0) THEN
          I12NUM = (MIN(I1SM,I2SM)-1)*NSMOB+MAX(I1SM,I2SM)
c          ELSE
c           I12NUM =  I1SM*(I1SM-1)/2+I2SM
c          END IF
          I12DIA = 0
          I34DIA = 0
          IF(IS12.EQ.0.OR.I1SM.NE.I2SM) THEN
            N12 = NO1PS(I1SM)*NO2PS(I2SM)
            I12DIA=0
          ELSE IF(IS12.EQ.1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
            I12DIA=1
          ELSE IF(IS12.EQ.-1.AND.I1SM.EQ.I2SM) THEN
            N12 = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
            I12DIA=-1
          END IF
          DO I3SM = 1, NSMOB
            I4SM = ADSXA(I3SM,I34SM)
            IF(I4SM.EQ.0) CYCLE
            IF(IS34.NE.0.AND.I3SM.GT.I4SM) CYCLE
c            IF(IS34.EQ.0) THEN
            I34NUM = (MIN(I3SM,I4SM)-1)*NSMOB+MAX(I3SM,I4SM)
c            ELSE
c             I34NUM =  I3SM*(I3SM-1)/2+I4SM
c            END IF
            IF(IS1234.NE.0.AND.I12NUM.GT.I34NUM) CYCLE
            IF(IS34.EQ.0.OR.I3SM.NE.I4SM) THEN
              N34 = NO3PS(I3SM)*NO4PS(I4SM)
              I34DIA=0
            ELSE IF(IS34.EQ.1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
              I34DIA=1
            ELSE IF(IS34.EQ.-1.AND.I3SM.EQ.I4SM) THEN
              N34 = NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
              I34DIA=-1
            END IF
            IF(IS1234.EQ.0.OR.I12NUM.NE.I34NUM) THEN
              IF (IMODE.NE.0) THEN
                IPNTR(I1SM,I2SM,I3SM) = IOFF
                ISM4A(I1SM,I2SM,I3SM) = I4SM
              END IF
              IF (I12DIA.EQ.I34DIA) THEN
                IOFF= IOFF+ N12 * N34
              ELSE IF (IS12.EQ.1.AND.IS34.EQ.0) THEN
                IF (I3SM.NE.I4SM) STOP 'non-covered exception'
                N12OOD=NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
                N12DIA=NO1PS(I1SM)
                N34LOW=NO3PS(I3SM)*(NO3PS(I3SM)+1)/2
                IOFF = IOFF+ N12OOD*N34 + N12DIA*N34LOW
              ELSE IF (IS12.EQ.0.AND.IS34.EQ.1) THEN
                IF (I1SM.NE.I2SM) STOP 'non-covered exception'
                N34OOD=NO3PS(I3SM)*(NO3PS(I3SM)-1)/2
                N34DIA=NO3PS(I3SM)
                N12LOW=NO1PS(I1SM)*(NO1PS(I1SM)+1)/2
                IOFF = IOFF+ N34OOD*N12 + N34DIA*N12LOW
              ELSE
                STOP 'non-covered case'
              END IF
            ELSE IF( IS1234.EQ.1.AND.I12NUM.EQ.I34NUM) THEN
              IF (IMODE.NE.0) THEN
                IPNTR(I1SM,I2SM,I3SM) = IOFF
                ISM4A(I1SM,I2SM,I3SM) = I4SM
              END IF
              IF (I12DIA.EQ.I34DIA) THEN
                IOFF= IOFF + N12*(N12+1)/2
              ELSE
                NN = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
                IF (ABS(I12DIA).EQ.1) MM=N12
                IF (ABS(I34DIA).EQ.1) MM=N34
                IOFF = IOFF + MM*(MM+1)/2 + NN*(NN+1)/2
              END IF
            ELSE IF (IS12.NE.0.AND.IS34.EQ.0.AND.IS1234.NE.0.AND.
     &               I1SM.EQ.I4SM.AND.I2SM.EQ.I3SM.AND.
     &               I12DIA.EQ.0.AND.I34DIA.EQ.0) THEN
              IF (IMODE.NE.0) THEN
                IPNTR(I1SM,I2SM,I3SM) = IOFF
                ISM4A(I1SM,I2SM,I3SM) = I4SM
              END IF
              IOFF = IOFF + N12*(N12+1)/2
            ELSE IF( IS1234.EQ.-1.AND.I12NUM.EQ.I34NUM) THEN
              IF (IMODE.NE.0) THEN
                IPNTR(I1SM,I2SM,I3SM) = IOFF
                ISM4A(I1SM,I2SM,I3SM) = I4SM
              END IF
              IF (I12DIA.EQ.I34DIA) THEN
                IOFF=  IOFF+ N12*(N12-1)/2
              ELSE 
                NN = NO1PS(I1SM)*(NO1PS(I1SM)-1)/2
                IF (ABS(I12DIA).EQ.1) MM=N12
                IF (ABS(I34DIA).EQ.1) MM=N34
                IOFF = IOFF + MM*(MM-1)/2 + NN*(NN+1)/2
              END IF
            END IF
C?          WRITE(6,*) ' I1SM I2SM I3SM I4SM    IOFF'
C?          WRITE(6,'(1H ,4I4,I9)')   I1SM,I2SM,I3SM,I4SM,IOFF

          END DO
        END DO
      END DO
*
*
      NEL = IOFF-1
C?    WRITE(6,*) ' PNT4DM , 64 elemets of IPNTR '
C?    call IWRTMA(IPNTR,1,64,1,64)
      NTEST = 0
      IF(NTEST.NE.0) THEN
         WRITE(6,*) ' Length of 4 index array ', IOFF - 1
      END IF
*
      RETURN
      END
      SUBROUTINE PRSM2(A,NDIM)
C
C PRINT LOWER TRIANGULAR MATRIX PACKED IN COLUMN WISE FASHION
C
      IMPLICIT REAL*8           ( A-H,O-Z)
      DIMENSION A(1)
C
      DO 100 I=1,NDIM
        WRITE(6,1010) I,
     &  (A((J-1)*NDIM-J*(J-1)/2+I),J=1,I)
  100 CONTINUE
      RETURN
 1010 FORMAT(1H0,2X,I3,5(1X,E13.7),/,(1H ,5X,5(1X,E13.7)))
      END
      SUBROUTINE PRTITL(LINES)
*
* Print title cards
*
      CHARACTER*102 LINES
      DIMENSION LINES(3)
      CHARACTER*80 STARS
*
      STARS(1:1) = ' '
      DO 80 I = 2, 80
        STARS(I:I) = '*'
   80 CONTINUE
      WRITE(6,'(A)') STARS
      WRITE(6,'(A1,A72,A1)') ' *  ',LINES(1),'  *'
      WRITE(6,'(A1,A72,A1)') ' *  ',LINES(2),'  *'
      WRITE(6,'(A1,A72,A1)') ' *  ',LINES(3),'  *'
      WRITE(6,'(A)') STARS
*
      RETURN
      END
      SUBROUTINE PSTTBL(C,CTT,IATP,IASM,IBTP,IBSM,IOCOC,
     &                  NOCTPA,NOCTPB,NSASO,NSBSO,PSIGN,
     &                  ICOOSC,IAC,IDC,LUHC,SCR,NSMST)
*
* add(IAC = 1) or copy (IAC =2) determinant block (iatp iasm, ibtp ibsm
* to vector packed in combination format
* iatp,iasm , ibtp,ibsm is assumed to be allowed combination block
*
* Combination type is defined by IDC
* IAC = 2  does not work for LUHC.NE.0 !
*. Note ICOOSC has been eliminated and the routine is 
*. therefore not working for LUHC = 0.
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION C(*),CTT(*),NSASO(NSMST, *),NSBSO(NSMST, *)
      DIMENSION IOCOC(NOCTPA,NOCTPB),ICOOSC(NOCTPA,NOCTPB,*)
*
      DIMENSION SCR(*)
*
* ======================
* Write directly to disc
* ======================
*
*. Assumes complete block in,
*. copies to lower half, scales  and write out.
      IF(LUHC.LE.0) THEN
        WRITE(6,*) 'PSTTBL, LUHC .le. 0 '
        WRITE(6,*) ' Update routine to eliminate ICOOSC'
        STOP'PSTTBL: Update routine to eliminate ICOOSC'
      END IF
*
      IF(LUHC.NE.0) THEN
         NAST = NSASO(IASM,IATP)
         NBST = NSBSO(IBSM,IBTP) 
         SCLFAC = 1.0D0
         PLSIGN = 1.0D0
         ISGVST = 1
         CALL SDCMRF(CTT,SCR,1,IATP,IBTP,IASM,IBSM,NAST,NBST,
     &               IDC,PSIGN,PLSIGN,ISGVST,LDET,LCOMB,
     &               1,SCLFAC)
*. Note: PLSIGN and ISGVST missing in order to make it work for IDC=3,4
         CALL ITODS(LCOMB,1,-1,LUHC)
         CALL TODSC(SCR,LCOMB,-1,LUHC)
      ELSE
* ==================
* Add to packed list
* ===================
      IF(IASM.GT.IBSM.OR.IDC.EQ.1
     &   .OR.IDC.EQ.3)THEN
**************
** IASM > IBSM
**************
 
        IF( IDC .LT. 4 ) THEN
*.. simple copying
C         IBASE = ICOOSC(IBTP,IATP,IASM)
          IBASE = ICOOSC(IATP,IBTP,IASM)
          NELMNT = NSASO(IASM,IATP)*NSBSO(IBSM,IBTP)
          IF(IAC .EQ. 1 ) THEN
            CALL VECSUM(C(IBASE),C(IBASE),CTT,1.0D0,1.0D0,NELMNT)
          ELSE IF(IAC .EQ.2 ) THEN
            CALL COPVEC(CTT,C(IBASE),NELMNT)
          END IF
        ELSE IF ( IDC .EQ. 4 ) THEN
          IF(IATP.GT.IBTP) THEN
C           IBASE = ICOOSC(IBTP,IATP,IASM)
            IBASE = ICOOSC(IATP,IBTP,IASM)
            NELMNT = NSASO(IASM,IATP)*NSBSO(IBSM,IBTP)
            IF(IAC .EQ. 1 ) THEN
              CALL VECSUM(C(IBASE),C(IBASE),CTT,1.0D0,1.0D0,NELMNT)
            ELSE IF(IAC .EQ.2 ) THEN
              CALL COPVEC(CTT,C(IBASE),NELMNT)
            END IF
          ELSE IF( IATP .EQ. IBTP ) THEN
C           IBASE = ICOOSC(IBTP,IATP,IASM)
            IBASE = ICOOSC(IATP,IBTP,IASM)
            NAST = NSASO(IASM,IATP)
            IF( IAC .EQ. 1 ) THEN
              CALL PMPLFM(C(IBASE),CTT,NDIM)
            ELSE
              CALL TRIPK3(CTT,C(IBASE),1,NAST,NAST,PSIGN)
            END IF
          END IF
        END IF
      ELSE IF( IASM.EQ.IBSM) THEN
**************
** IASM = IBSM
**************
        IF(IATP.GT.IBTP) THEN
*.. simple copying
C         IBASE = ICOOSC(IBTP,IATP,IASM)
          IBASE = ICOOSC(IATP,IBTP,IASM)
          NELMNT = NSASO(IASM,IATP)*NSBSO(IBSM,IBTP)
          IF(IAC .EQ. 1 ) THEN
            CALL VECSUM(C(IBASE),C(IBASE),CTT,1.0D0,1.0D0,NELMNT)
          ELSE IF(IAC .EQ.2 ) THEN
            CALL COPVEC(CTT,C(IBASE),NELMNT)
          END IF
        ELSE IF( IATP.EQ.IBTP) THEN
*.. reform to triangular packed matrix
C         IBASE = ICOOSC(IBTP,IATP,IASM)
          IBASE = ICOOSC(IATP,IBTP,IASM)
          NAST = NSASO(IASM,IATP)
          IF( IAC .EQ. 1 ) THEN
            CALL PMPLFM(C(IBASE),CTT,NAST)
          ELSE
            CALL TRIPK3(CTT,C(IBASE),1,NAST,NAST,PSIGN)
          END IF
        END IF
      END IF
      END IF
*
      RETURN
      END
      SUBROUTINE PUTREC(LU,LBLK,REC,LENGTH)
*
* PUT RECORD ON FILE LU
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION REC(*)
*
      IF(LBLK .GT. 0 ) THEN
        LENGTH = LBLK
        CALL TODSC(REC,LENGTH,LENGTH,LU)
      ELSE
        CALL ITODS(LENGTH,1,LBLK,LU)
        CALL TODSC(REC,LENGTH,LBLK,LU)
      END IF
*
      RETURN
      END
      SUBROUTINE RAS3DF_OLD(IFLAG,NINASH,NDELSH,NRSSH,NIRREP)
*
* Iflag = 1: Define RAS3 as the orbitals not explicitly made 
*             inactive, deleted, ras1,ras2
*  
* Iflag = 2: Define Deleted  as the orbitals not explicitly made 
*             inactive, deleted, ras1,ras2,ras3
* Obtain default values for occupation in RAS 3 as
* the orbitals not explicitly made inactive, deleted, RAS1,RAS2.
* Total number of oribtals obtained from information in /MOLOBS/ as
* obtained in GETOBS
*
       IMPLICIT REAL*8           (A-H,O-Z)
       INCLUDE 'mxpdim.inc'
      COMMON/MOLOBS/
     & IOList(20),iToc(64),nBas(8),nOrb(8),nFro(8),nDel(8),
     & Nsym
*
      DIMENSION NINASH(MXPIRR),NDELSH(MXPIRR),NRSSH(MXPIRR,3)
*
      DO 100 ISM = 1, NIRREP
        IF( IFLAG.EQ.1) THEN
          NRSSH(ISM,3) = NORB(ISM)-NINASH(ISM)-NDELSH(ISM)
     &                 - NRSSH(ISM,1)-NRSSH(ISM,2)
        ELSE IF(IFLAG.EQ.2) THEN
          NDELSH(ISM) =  NORB(ISM)-NINASH(ISM)
     &                 - NRSSH(ISM,1)-NRSSH(ISM,2)
     &                 - NRSSH(ISM,3)
        END IF
  100 CONTINUE
*
      NTEST = 1
      IF(NTEST.NE.0) THEN
        IF(IFLAG.EQ.1) THEN
          WRITE(6,*) 
     &    ' Number of orbitals in RAS 3 as supplied from RAS3DF'
          CALL IWRTMA(NRSSH(1,3),1,NIRREP,1,NIRREP)
        ELSE IF(IFLAG.EQ.2) THEN
          WRITE(6,*) 
     &    ' Number of orbitals in DELETED as supplied from RAS3DF'
          CALL IWRTMA(NDELSH,1,NIRREP,1,NIRREP)
        END IF
      END IF
*
      RETURN
      END
      SUBROUTINE RASSG3(CB,SB,NBATS,LBATS,LEBATS,I1BATS,IBATS,
     &           LUC,LUHC,CV,SV,ECORE,ITASK)
*
* Direct RAS routine employing combined MOC/n-1 resolution method
*
* Jeppe Olsen   Winter of 1991
*               May 1997: Connected to SBLOCK
*               May 2010: Argument changed to allow in core (ICISTR=1)..
*               Jan. 2012: If Logical unit numbers are negative, the 
*                          complete C and S vectors are/will be in SV,CV
*               March 2012: ITASK added
*
* =====
* Input
* =====
*

      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cprnt.inc'
*. Batches of sigma
      INTEGER LBATS(*),LEBATS(*),I1BATS(*),IBATS(8,*)
*.Scratch
      DIMENSION SB(*),CB(*)
*. Input/output if ICISTR = 1
      DIMENSION SV(*),CV(*)
      CHARACTER*6 ITASK
*
      CALL QENTER('RASSG')
      NTEST = 000
      NTEST = MAX(NTEST,IPRCIX)
      IF(NTEST.GE.20) THEN
        WRITE(6,*) ' ================='
        WRITE(6,*) ' RASSG3 speaking:'
        WRITE(6,*) ' ================='
        WRITE(6,*) ' RASSG3: NBATS = ',NBATS
        WRITE(6,'(A,A6)') ' ITASK = ', ITASK
      END IF
*
      IF(LUHC.GT.0) CALL REWINO(LUHC)
* Loop over batches over sigma blocks
      IOFF_S = 1
      DO JBATS = 1, NBATS
        IF(NTEST.GE.10000) WRITE(6,*) ' JBATS, LBATS(..) = ',
     &                                  JBATS, LBATS(JBATS)
*. Obtain sigma for batch of blocks
C            SBLOCK(NBLOCK,IBLOCK,IBOFF,CB,HCB,LUC,IRESTRICT,LUCBLK) 
        CALL SBLOCK(LBATS(JBATS),IBATS(1,I1BATS(JBATS)),1,
     &       CB,SB,LUC,0,0,0,0,0,CV,ECORE,ITASK)
*. Transfer S block to permanent storage
        DO ISBLK = I1BATS(JBATS),I1BATS(JBATS)+ LBATS(JBATS)-1
          IATP = IBATS(1,ISBLK)
          IBTP = IBATS(2,ISBLK)
          IASM = IBATS(3,ISBLK)
          IBSM = IBATS(4,ISBLK)
          IOFF = IBATS(6,ISBLK)
          LEN  = IBATS(8,ISBLK)
C?        write(6,*) 'RASSG3: IOFF, SB(IOFF)',IOFF,SB(IOFF)
          IF(ICISTR.NE.1) THEN
            CALL ITODS(LEN,1,-1,LUHC)
            CALL TODSC(SB(IOFF),LEN,-1,LUHC)
          ELSE
            CALL COPVEC(SB(IOFF),SV(IOFF_S),LEN)
            IOFF_S = IOFF_S + LEN
          END IF
        END DO
      END DO
*
      IF(ICISTR.NE.1) CALL ITODS(-1,1,-1,LUHC)
      IF(NTEST.GE.100) THEN
        IF(ICISTR.NE.1) THEN
          WRITE(6,*) ' Final S-vector on disc'
          CALL WRTVCD(SB,LUHC,1,-1)
        ELSE
          LEN_S = IOFF_S - 1
          WRITE(6,*) ' Final S-vector'
          CALL WRTMAT(SV,1,LEN_S,1,LEN_S)
        END IF
      END IF
      IF(NTEST.GE.100) WRITE(6,*) ' Leaving RASSG3'
*
      CALL QEXIT('RASSG')
      RETURN
      END
      SUBROUTINE READIN(LUIN,LUOUT)
*
*
* File is supposed to be positioned at first record of input
* The end of the input stream is identified by END OF INPUT
* Unless MOLCS is specified,
* All keywords are initiated by a point ., while comments are
* initiated by a *.
*
* The keywords are can broadly be divided into two types
*  1: Keywords describing CI calculation to be carried out
*  2: Keywords describing how CI optimization should be performed
*
*
* All input parameter concerning CI space are saved in /LUCIN1/
* All input concerning actual CI vectors are save in /CSTATE/
* All input paramters concerning run are saved in /CRUN/
*
* Since the keywords are read in from one pass over input file,
* the keywords must be in logical order.For example, the number
* of irreducible representations (irreps) must be give before
* the number of shells per irrep
*
* Jeppe Olsen, Initiated spring of 1991
*
*. Last modification; Jeppe Olsen; July 8, 2013; Revamping EXPHAM
*
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
*
COLD  PARAMETER(MXPLNC = 72 )
      CHARACTER*102 TITLEC
      CHARACTER*101 CARD
      CHARACTER*102 CARD1
      CHARACTER*102 LASTCARD
      CHARACTER*10 CARDX
      COMMON/CTITLE/ TITLEC(3)
      CHARACTER*6 KEYWOR
      PARAMETER(MXPKW = 202)
      DIMENSION KEYWOR(MXPKW)
      DIMENSION ISETKW(MXPKW)
*. Local  scratch for decoding multi-item lines, atmost 32 items per line
      PARAMETER(MX_ITEM = 40)
      CHARACTER*102 ITEM(MX_ITEM),  ITEMX 
      INTEGER INT_ITEM(MX_ITEM)
*
      DATA KEYWOR/'TITLE ','PNTGRP','NIRREP','INTSPC','EXTSPC',
     &            'NACTEL','INACT ','CORE  ','RAS1  ','RAS2  ',
     &            'RAS3  ','MXSCTP','SECOND','REFSPC','INTSEL',
     &            'MS2   ','MULTS ','IREFSM','ROOTS ','IDIAG ',
     &            'MAXIT ','EXPHAM','RESTRT','INTIMP','INCORE',
     &            'DELETE','MSCOMB','MLCOMB','IPRSTR','IPRCIX',
     &            'IPRORB','IPRDIA','MXCIV ','CISTOR','NOCSF ',
     &            'IPRXT ','NOINT ','DMPINT','RESDIM','CJKAIB',
     &            'INIREF','RESTRF','IPROCC','MOCAA ','MOCAB ',
     &            'ECORE ','PERTU ','APRREF','APRZER','GASSH ',
     &            'GASSPC','CMBSPC','CICONV','SEQUEN','EXTKOP',
     &            'MACHIN','C1DSC ','H0SPC ','H0FRM ','RFROOT',
     &            'H0EX  ','INIDEG','LAMBDA','LCSBLK','IPRDEN',
     &            'NOMOFL','ECHO  ','FINORB','E_THRE','C_THRE',
     &            'E_CONV','C_CONV','CLSSEL','DENSI ','PTEKT ',
     &            'H0ROOT','NORST2','SKIPEI','XYZSYM','PROPER',
     &            'TRAPRP','RESPON','MXITLE','IPRRSP','RTHOME',
     &            'USE_PH','ADVICE','TRACI ','USE_PS','PTFOCK',
     &            'PRNCIV','RES_CC','TRA_FI','TRA_IN','MUL_SP',
     &            'RELAX ','EXPERT','CNV_RT','IPRPRO','QDOTS ',
     &            'RE_MS2','PRECON','SIMSYM','USE_HW','USEH0P',
     &            'IPRCC ','LZ2   ','CCSOLV','CCN   ','SBSPJA',
     &            'CCCONV','NHOSPC','CC3   ','CI=>CC','CCFORM',
     &            'CCEX_E','RES_EX','RESDCC','CMB_CC','SIMTRH',
     &            'FRZ_CC','CC_EXP','OLDCCV','NEWCCP','MXSPOX',
     &            'MASKSD','NOAAEX','SPINRS','GENTRD','REO_OR',
     &            'IC_EXC','CMPCCI','CC=>CI','COMHAM','DMPMRP',
     &            'VNEWCC','OLDCCV','HF_INI','HFSOLV','SPNDEN',
     &            'GTBOPT','GTBEAG','GTBFOO','GTBHSS','WRKSPC',
     &            'SAVMEM','TARGET','CUMULA','RSTRIC','NCOMMU',
     &            'APRCME','APRCMV','APRCMJ','DENSPC','READSJ',
     &            'PRDEXP','PRDWVF','PRDEXC','IPRCSF','INC_AA',
     &            'SINGU ','MXVC_I','MXIT_M','FR_INT','ZS_HAM',
     &            'NO_EI ','IC_EXO','GIC_EX','IPRMCS','MCSCFA',
     &            'TRA_RO','NORTIN','VBRFSP','NORT_M','MOFRAG',
     &            'FRAGOB','INI_MO','INICNF','PRVB'  ,'GIOMET',
     &            'IPRINT','H0_CSF','CNFBAT','ENSMGS','ENSCON',
     &            'EQFRAG','SUPSYM','HFD_OC','HFS_OC','GAS_SP',
     &            'NOSPFI','FRG=LU','FRZORB','SBSPPR','IN_NRO',
     &            'IN_SRO','RT_SEL','FRZFST','VBGNSP','VBOBOR',
     &            'VBSCOR','AKBKME'/
*
COLD  INTEGER CITYP
*.Largest allowed number of allowed irreps for orbs
 
      INCLUDE 'lucinp.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'crun.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'machine.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'cecore.inc'
      INCLUDE 'newccp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'gtbce.inc'
      INCLUDE 'opti.inc'
      INCLUDE 'symrdc.inc'
      INCLUDE 'prdwvf.inc'
      INCLUDE 'vb.inc'
      INCLUDE 'fragmol.inc'
      INCLUDE 'comjep.inc'
*. Flag for compatibility with normal MOLCAS input format
      MOLCS = 1
*
      IEXPERT = 0
      NERROR = 0
      NWARN = 0
      EXTSPC = 0
      IECHO = 0
      I_DO_MCSCF = 0
      I_DO_HF = 0
      I_DO_NORTCI = 0
      I_DO_NORTMCSCF = 0
      I_DO_GAS = 0
      ISPNDEN = 0
* No cc as default
      I_DO_CC = 0
      I_DO_GTBCE = 0
      I_DO_ICCC = 0
*. Start out with normal integrals
      I_USE_SIMTRH = 0
      I_UNRORB = 0
      ISPCAS = 0
*. I do not do EI unless I am told to do it...
      I_DO_EI = 0
*. Default is no fragments
      NFRAG_MOL = 0 
*
      I_DO_GIC = 0
*. Stupid compiler warning 
      ICI = -3006
      ISECFILL = 0

*****************************************************************
*                                                               *
* Part 1: Read in Keywords and perform some preliminary checks *
*                                                               *
*****************************************************************
*
*. Defaults for pointgroup and number of irreps must be set here
*. Default point group D2H
      PNTGRP = 1
*. Default number of irreps
      NIRREP = 8
*
      DO ISM = 1, NIRREP
        NTOOBS(ISM) = 0
      END DO
*. Largest allowd number of IRREPS for super-symmetry
      MAX_SUPSYM_IRREP = 2*MXPL + 1
      NACT_SUPSYM_IRREP = 0
*
      CALL ISETVC(ISETKW,0,MXPKW)
 1000 CONTINUE
*. Next potential keyword
        READ(LUIN,'(A)') CARD
*. Left-position nonblank characters in CARD
        CALL LFTPOS(CARD,MXPLNC)
*. Change to upper case
C            UPPCAS(LINE,LENGTH)
        CALL UPPCAS(CARD,MXPLNC)
        IF(CARD(1:1).EQ.'*'.OR.CARD(1:1).EQ.'!'.OR.
     &     CARD(1:1).EQ.'#'                        )THEN
*. Skip comment cards
          GOTO 999
*. End of input card
        ELSE IF(CARD(1:5).EQ.'ENDOF'.OR.CARD(1:6).EQ.'END OF') THEN
          GOTO 1001
        ELSE IF(MOLCS.EQ.0.AND.CARD(1:1).NE.'.') THEN
*. Line out of context
          WRITE(LUOUT,'(1H ,A)') ' Warning, card out of context: '
          WRITE(LUOUT,'(1H ,A)') CARD
          NWARN = NWARN + 1
        ELSE IF(MOLCS.EQ.1.OR.CARD(1:1).EQ.'.') THEN
          IF(MOLCS.EQ.1) THEN
*. Move characters one place to right
          DO 1286 ICHAR = 7,2,-1
            CARD(ICHAR:ICHAR) = CARD(ICHAR-1:ICHAR-1)
 1286       CONTINUE
            CARD(1:1) = ' '
          END IF
*. A keyword has been identified, match with possible keywords
          IF(CARD(2:6).EQ.'TITLE' ) THEN
*
* =========================
*.Keyword 1:  TITLE cards
* =========================
*
*. Three title cards
            ISETKW(1) = 1
            DO 20 IC = 1, 3
              READ(LUIN,'(A)') TITLEC(IC)
   20       CONTINUE
            GOTO 999
          END IF
*
*
*================================================
*. Keyword 2: <POINTG>: Point group of orbitals
*================================================
*
* Possible point groups: D2H,CINFV,DINFH,O3
          IF(CARD(2:4).EQ.'D2H'   .OR.
     &       CARD(2:6).EQ.'CINFV' .OR.
     &       CARD(2:6).EQ.'DINFH' .OR.
     &       CARD(2:3).EQ.'O3'    ) THEN
*
            ISETKW(2) = 1
            IF(CARD(2:4).EQ.'D2H') THEN
              PNTGRP = 1
            ELSE IF(CARD(2:6).EQ.'CINFV') THEN
              PNTGRP = 2
            ELSE IF(CARD(2:6).EQ.'DINFH') THEN
              PNTGRP = 3
            ELSE IF(CARD(2:3).EQ.'O3') THEN
              PNTGRP = 4
            END IF
            GOTO 999
          END IF
*
          IF(CARD(2:7).EQ.'NIRREP') THEN
*
*=====================================================
*. Keyword 3: <NIRREP>: Number of irreps of orbitals
*=====================================================
*
* Number of irreducible representations in point group
* D2h             : 1,2,4,8
* C inf H, D inf H: largest ML
* O3              : Largest L
*.D2h or subgroup
* ===============
            IF(PNTGRP.EQ.1) THEN
              READ(LUIN,*) NIRREP
              NSMCMP = NIRREP
              NSMOB  = NIRREP
              ISETKW(3) = 1
*.Dimensions 3,5,6,7,8 are not allowed
              IF(NIRREP.EQ.3.OR.(NIRREP.GT.4.AND.NIRREP.LT.8)) THEN
                 WRITE(LUOUT,*) ' Input error: NIRREP = ', NIRREP
                 WRITE(LUOUT,*) ' Allowed values of NIRREP:1,2,4,8'
                 NERROR = NERROR + 1
                 ISETKW(3) = -1
              END IF
*. Zero values used for other pointgroups
COLD          MAXML  = -1
COLD          MAXL   = -1
COLD          INVCNT = -1
            ELSE IF (PNTGRP.EQ.2) THEN
*. Cinf V
* =======
              READ(LUIN,*) MAXML
              ISETKW (3) = 1
              IF(MAXML.LT.0) THEN
                WRITE(LUOUT,*)
     &          ' Largest ML values of shells must be atleast be zero '
                WRITE(LUOUT,*) ' MAXML from input:' ,MAXML
                NERROR = NERROR + 1
                ISETKW(3) = -1
              END IF
              NIRREP =  MAXML + 1
              NSMCMP = 2 * MAXML + 1
              NSMOB = NSMCMP
COLD          INVCNT = 0
              MAXL = -1
            ELSE IF (PNTGRP.EQ.3) THEN
*. Dinf H
* =======
              READ(LUIN,*) MAXML
              ISETKW (3) = 1
              IF(MAXML.LT.0) THEN
                WRITE(LUOUT,*)
     &          ' Largest ML values of shells must be atleast be zero '
                WRITE(LUOUT,*) ' MAXML from input: ',MAXML
                NERROR = NERROR + 1
                ISETKW(3) = -1
              END IF
              NIRREP = 2 * ( MAXML + 1)
              NSMCMP = 2 * ( 2*MAXML + 1 )
              NSMOB = NSMCMP
COLD          INVCNT = 1
              MAXL  = -1
            ELSE IF (PNTGRP.EQ.4) THEN
*. O 3
* =======
              READ(LUIN,*) MAXL
              ISETKW (3) = 1
              IF(MAXL.LT.0) THEN
                WRITE(LUOUT,*)
     &          ' Largest L values of shells must be atleast be zero '
                WRITE(LUOUT,*) ' MAXL from input: ' , MAXL
                NERROR = NERROR + 1
                ISETKW(3) = -1
              END IF
              MAXML = MAXL
              NIRREP = MAXL + 1
              NSMCOM = 2 * (2 *MAXML + 1 )
              NSMOB = NSMCMP
COLD          INVCNT = 1
            END IF
            IF(ISETKW(3).EQ.-1)
     &      WRITE(LUOUT,*) ' .NIRREP input incorrect !! . '
            GOTO 999
          END IF
*
* ================================================
*. Keyword 4: INTSPC: Type of internal CI space
* ================================================
*
          IF(CARD(2:4).EQ.'CAS'.OR.CARD(2:4).EQ.'FCI'.OR.
     &       (CARD(2:4).EQ.'RAS'.AND.CARD(5:5).EQ.' ')) THEN
            ISETKW(4) = 1
            IF(CARD(2:4).EQ.'CAS'.OR. CARD(2:4).EQ.'FCI' ) THEN
              INTSPC = 1
            ELSE IF (CARD(2:4).EQ.'RAS') THEN
              INTSPC = 2
*. Limits on allowed number of electrons in RASI and RAS III
              READ(LUIN,*) MNRS10,MXRS30
            END IF
            GOTO 999
          END IF
*
*===========================
* Keyword 5: External space
*===========================
*
         IF(CARD(2:7).EQ.'EXTSPC') THEN
           ISETKW(5) = 1
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
           IF(CARD1(1:4).EQ.'NONE') THEN
             EXTSPC = 0
             MXER4 = 0
             MXHR0 = 0
           ELSE IF
     &     (CARD1(1:4).EQ.'CORE'.AND.CARD1(5:10).EQ.'SECOND') THEN
             EXTSPC = 3
             READ(LUIN,*) MXHR0,MXER4
           ELSE IF
     &     (CARD1(1:6).EQ.'SECOND'.AND.CARD1(7:10).EQ.'CORE') THEN
             EXTSPC = 3
             READ(LUIN,*) MXER4,MXHR0
           ELSE IF(CARD1(1:4).EQ.'CORE') THEN
             EXTSPC = 1
             READ(LUIN,*) MXHR0
           ELSE IF(CARD1(1:6).EQ.'SECOND') THEN
             EXTSPC = 2
             READ(LUIN,*) MXER4
           ELSE
             ISETKW(5) = - 1
             WRITE(6,*) ' Illegal card for EXTSPC: '
             WRITE(6,'(1H ,A)') CARD1
             NERROR = NERROR + 1
           END IF
           GOTO 999
         END IF
*
* =============================================
* Keyword 6 NACTEL: Number of active electrons
* =============================================
*
         IF(CARD(2:7).EQ.'NACTEL') THEN
           READ(LUIN,*)NACTEL
           ISETKW(6) = 1
           IF(NACTEL.LT.0) THEN
             WRITE(LUOUT,*)
     &       ' ERROR: Illegal number of active electrons ', NACTEL
             ISETKW(6) = -1
             NERROR = NERROR + 1
           END IF
           GOTO 999
         END IF
*==================
* 7: Inactive shells
*==================
         IF(CARD(2:7).EQ.'INACTI'.OR.CARD(2:6).EQ.'INASH') THEN
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
*. A line can be one of the following 
*  NIRREP numbers giving dim of each irrep for this space
* A character entry:
*                     NONE => No orbitals in this space
          CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          ITEMX = ITEM(1)
          IF(ITEMX(1:4).EQ.'NONE') THEN
            DO IRREP = 1, NIRREP
              NINASH(IRREP) = 0
            END DO
          ELSE 
*. I expect that NIRREP integers are given
            IF(NITEM.NE.NIRREP) THEN
              WRITE(6,*) ' Erroneous input to INASH: '
              WRITE(6,'(72A)') CARD1
              WRITE(6,*) ' Specify either:   NONE '
              WRITE(6,*) ' Or NIRREP integers  '
              NERROR = NERROR + 1
              ISETKW(7) = -1
            END IF
*. Well assume NIRREP integers
            DO IRREP = 1, NIRREP
              CALL CHAR_TO_INTEGER(ITEM(IRREP),NINASH(IRREP),
     &             MXPLNC)
            END DO
          END IF
*.  Update number of orbitals per symmetry
          DO IRREP = 1, NIRREP
            NTOOBS(IRREP) = NTOOBS(IRREP) + NINASH(IRREP)
          END DO
          ISETKW(7) = 1
          GOTO 999
         END IF
*=================================
* 8: Core shells ( = RAS0 shells)
*==================================
         IF(CARD(2:5).EQ.'CORE') THEN
           READ(LUIN,*) (NRS0SH(1,IRREP),IRREP = 1, NIRREP)
           ISETKW(8) = 1
           EXTSPC = EXTSPC + 1
           GOTO 999
         END IF
*===========
* 9: RAS 1
*===========
         IF(CARD(2:5).EQ.'RAS1') THEN
*.Number of RAS 1 shells per irrep
           READ(LUIN,*) (NRSSH(IRREP,1),IRREP = 1, NIRREP)
*.Smallest allowed number of electrons in RAS 1
C!         READ(LUIN,*) MNER10
           ISETKW(9) = 1
           GOTO 999
         END IF
*===========
* 10: RAS 2
*===========
         IF(CARD(2:5).EQ.'RAS2'.OR.CARD(2:7).EQ.'ACTIVE') THEN
           READ(LUIN,*) (NRSSH(IRREP,2),IRREP = 1, NIRREP)
           ISETKW(10) = 1
           GOTO 999
         END IF
*===========
* 11: RAS 3
*===========
         IF(CARD(2:5).EQ.'RAS3') THEN
           ISETKW(11) = 1
*.Number of RAS 3 shells per irrep
           READ(LUIN,*) (NRSSH(IRREP,3),IRREP = 1, NIRREP)
*.Largest allowed number of electrons in RAS III
C!         READ(LUIN,*) MXER30
           GOTO 999
         END IF
* ==================================================
* 13: Number of shells in secondary space per type
* =================================================
         IF(CARD(2:7).EQ.'SECOND'.OR.CARD(2:6).EQ.'SECSH') THEN
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
*. A line can be one of the following 
*  NIRREP numbers giving dim of each irrep for this space
* A character entry:
*                     NONE => No orbitals in this space
*                     ALL  => All remaining orbitals  in this space
*                     REST => All remaining orbitals  in this space
*. Note: Only a single space must be defined by ALL or REST
          CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          ITEMX = ITEM(1)
          IF(ITEMX(1:4).EQ.'NONE') THEN
            DO IRREP = 1, NIRREP
              NSECSH(IRREP) = 0
            END DO
          ELSE IF(ITEMX(1:3).EQ.'ALL'.OR.ITEMX(1:4).EQ.'REST') THEN
*. Only a single space must be defined in this way
            IF(IGSFILL.NE.0) THEN
              WRITE(6,*) 
     &        ' Several shell spaces defined by ALL or REST'
              WRITE(6,*)
     &        ' This confuses and upsets me '
              WRITE(6,*)
     &        '                                / Lucia '
              ISETKW(13) = -1
              NERROR = NERROR + 1
            END IF
            ISECFILL = 1
          ELSE 
*. I expect that NIRREP integers are given
            IF(NITEM.NE.NIRREP) THEN
              WRITE(6,*) ' Erroneous input to SECSH: '
              WRITE(6,'(72A)') CARD1
              WRITE(6,*) ' Specify either:   NONE '
              WRITE(6,*) '                     ALL' 
              WRITE(6,*) '                    REST' 
              WRITE(6,*) ' Or NIRREP integers  '
              NERROR = NERROR + 1
              ISETKW(50) = -1
            END IF
*. Well assume NIRREP integers
            DO IRREP = 1, NIRREP
              CALL CHAR_TO_INTEGER(ITEM(IRREP),NSECSH(IRREP),
     &             MXPLNC)
            END DO
          END IF
*.  Update number of orbitals per symmetry
          DO IRREP = 1, NIRREP
            NTOOBS(IRREP) = NTOOBS(IRREP) + NSECSH(IRREP)
          END DO
          ISETKW(13) = 1
          GOTO 999
         END IF
* =========================
* 14: Reference space
* =========================
*
* Reuse of old keyword, august 2002
*
* Three forms 
* Number of entries = Number of Gasspaces => Allowed occupation of each 
*                                            GAS pace
* Number of entries = 2 * Number of gasspaces => Accumulated occupation
* A single entry Auto is given, indicating automatic generation 
*
*
         IF(CARD(2:7).EQ.'REFSPC') THEN
*. Number of GAS paces, NGAS, must have been defined through keyword GASSH
          IF(ISETKW(50).EQ.0) THEN 
            WRITE(6,*) 
     &      ' READIN: Keyword GASSH must be specified before REFSPC '
            NERROR = NERROR + 1
            GOTO 999
            ISETKW(14) = -1
          END IF
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
          CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          IF(NITEM.EQ.1.AND.CARD1(1:1).EQ.'A') THEN
*. Automatic generation
*. Has orbital partitionings been specified ?. At the moment, 
*. only a version with two partiotionings are active in input
            ISETKW(14) = 1
          ELSE IF(NITEM.EQ.NGAS) THEN
*. Specification of occupations
            ISETKW(14) = 1
            DO IGAS = 1, NGAS
              CALL CHAR_TO_INTEGER(ITEM(IGAS),IREFOCC(IGAS),MXPLNC)
            END DO
*. Reform to accumulated occupations 
            NEL = 0
            DO IGAS = 1, NGAS
              NEL = NEL + IREFOCC(IGAS)
              IREFOCC_ACC(IGAS,1) = NEL
              IREFOCC_ACC(IGAS,2) = NEL
            END DO
          ELSE IF ( NITEM.EQ. 2*NGAS) THEN
            ISETKW(14) = 1
*. Accumulated occupations 
            J = 0
            DO IGAS = 1, NGAS
              DO IMNMX = 1, 2
                J = J + 1
                CALL CHAR_TO_INTEGER(ITEM(J),IREFOCC_ACC(IGAS,IMNMX),
     &               MXPLNC)
              END DO
            END DO
          ELSE
            ISETKW(14) = -1
            WRITE(6,*) 
     &      ' READIN: Wrong number of entries in REFPSC '
            WRITE(6,*) 
     &      '          Use NGAS numbers for occupation '
            WRITE(6,*) 
     &      '          or  2*NGAS numbers for acc. occupations'
            NERROR = NERROR + 1
          END IF
          GOTO 999
         END IF
*
* =========================================================
* 15: selection of active configurations in internal space
* =========================================================
*
         IF(CARD(2:7).EQ.'INTSEL' ) THEN
           ISETKW(15) = 1
*
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
*
           IF(CARD1(1:4).EQ.'NONE') THEN
*. All internals are included
             INTSEL = 0
           ELSE IF(CARD1(1:6).EQ.'INDTST') THEN
*. Include coeffcients larger than CTHRES or having energy contributions
*. larger than ETHREA
             INTSEL = 1
             READ(LUIN,*)  CTHRES,ETHRES
           ELSE IF(CARD1(1:6).EQ.'TOTTST') THEN
*. Obtain CTHRES of the total wavefinction and ETHRES of the total
*. energy
             INTSEL = 2
             READ(LUIN,*)  CTHRES,ETHRES
           ELSE IF(CARD1(1:6).EQ.'INDWCN') THEN
*. Include all configutations with reference weights larger than
*. a given threshold in reference CI
              INTSEL = 3
              READ(LUIN,*) XWCNF
           ELSE IF(CARD1(1:6).EQ.'TOTWCN') THEN
*. Include the largest configurations so all a given fraction
*. of the Zero order reference is included in the CI
             INTSEL = 4
             READ(LUIN,*) XWCNF
           ELSE
             WRITE(LUOUT,*) ' Error: Illegal card in INTSEL:'
             WRITE(LUOUT,'(1H ,A)') CARD1
             ISETKW(15) = - 1
             NERROR = NERROR + 1
           END IF
           GOTO 999
         END IF
*===============================
* 16: Two times spin projection
* ==============================
          IF(CARD(2:4).EQ.'MS2') THEN
            ISETKW(16) = 1
            READ(LUIN,*) MS2
            GOTO 999
          END IF
*========================
* 17: spin multiplicity
* =======================
          IF(CARD(2:6).EQ.'MULTS') THEN
            ISETKW(17) = 1
            READ(LUIN,*) MULTS
            GOTO 999
          END IF
*========================
* 18: Reference symmetry
* =======================
          IF(CARD(2:7).EQ.'IREFSM') THEN
            ISETKW(18) = 1
            IF(PNTGRP.EQ.1) THEN
               READ(LUIN,*) IREFSM
            ELSE IF(PNTGRP.EQ.2) THEN
               READ(LUIN,*) IREFML
            ELSE IF (PNTGRP.EQ.3) THEN
               READ(LUIN,*) IREFML,IREFPA
               IF(IREFPA.EQ.-1) IREFPA = 2
            ELSE IF (PNTGRP.EQ.3) THEN
               READ(LUIN,*) IREFL,IREFML,IREFPA
               IF(IREFPA.EQ.-1) IREFPA = 2
            END IF
            GOTO 999
          END IF
*==========================
* 19: Roots to be obtained
* =========================
          IF(CARD(2:6).EQ.'ROOTS') THEN
            ISETKW(19) = 1
            READ(LUIN,*) NROOT
            DO I = 1, NROOT
              IROOT(I) = I
            END DO
C           READ(LUIN,*) (IROOT(I),I=1,NROOT)
            GOTO 999
          END IF
*===============================
* 20: Diagonalization algorithm  : .MEGACI , .TERACI
*===============================
          IF(CARD(2:7).EQ.'MEGACI') THEN
            ISETKW(20) = 1
            IDIAG = 1
            GOTO 999
          ELSE IF(CARD(2:7).EQ.'TERACI') THEN
            ISETKW(20) = 1
            IDIAG = 2
            GOTO 999
          END IF
*==================================
* 21: Explicit hamilton matrix  : MXP1,MXP2,MXQ
*==================================
          IF(CARD(2:7).EQ.'EXPHAM') THEN
*. Construction of explicit Hamiltonian in subspace
*. Lines: 1: Method for selecting subspace, 
*         2: Allowed dimension or identification of 
*            subspace
            READ(LUIN,*) ISBSPC_SEL
* ISBSPC_SEL = 1: Choose lowest elements of diagonal
*            = 2: Choose first elements
*            = 3: Choose a CI-space
*            = 4: read in a minmax space
            IF(ISBSPC_SEL.EQ.1.OR.ISPSPC_SEL.EQ.2) THEN
*. Read in Dimension of lowest space (just a single space)
              READ(5,*) MXP1
              MXP2 = 0
              MXQ = 0 
            ELSE IF (ISBSPC_SEL.EQ.3) THEN
*. Read in the subspace to be used 
              READ(5,*) ISBPSC_SPC
            ELSE IF(ISBSPC_SEL.EQ.4) THEN
*. Read Min max occupation of subspace 
              READ(5,*) NSBSPC_ORB
              READ(5,*) ( ISBSPC_MINMAX(I,1),I=1, NSBSPC_ORB)
              READ(5,*) ( ISBSPC_MINMAX(I,2),I=1, NSBSPC_ORB)
*. And the active orbitals of the MINMAX 
              READ(5,*) ( ISBSPC_ORB(I),I=1, NSBSPC_ORB)
            ELSE
              WRITE(6,*) ' Unknown value of ISBSPC_SEL = ', ISBSPC_SEL
              NERROR = NERROR + 1
            END IF
COLD        READ(LUIN,*) MXP1,MXP2,MXQ
            ISETKW(21) = 1
            GOTO 999
          END IF
*===================================================
* 22: Largest allowed number of Iterations per root: MAXIT
*===================================================
          IF(CARD(2:6).EQ.'MAXIT') THEN
            ISETKW(22) = 1
            READ(LUIN,*) MAXIT
            GOTO 999
          END IF
*====================
* 23: Restart option
*====================
         IF(CARD(2:7).EQ.'RESTRT') THEN
           ISETKW(23) = 1
           IRESTR = 1
           GOTO 999
         END IF
*========================================
* 24 Import of integrals and environment
*========================================
        IF(CARD(2:7).EQ.'MOLCAS'.OR.CARD(2:7).EQ.'ENV=MO') THEN
*. Integrals imported from MOLCAS
          IDO_LIPKIN = 0
          INTIMP = 1
          ISETKW(24) = 1
          ENVIRO(1:6) = 'MOLCAS'
          GOTO 999
        ELSE IF(CARD(2:6).EQ.'LUCAS')THEN
*. Integrals imported from LUCAS
          IDO_LIPKIN = 0
          INTIMP = 2
          ISETKW(24) = 1
          ENVIRO(1:6) = 'LUCAS '
          GOTO 999
        ELSE IF(CARD(2:7).EQ.'FMINSM'.OR.CARD(2:7).EQ.'ENV=FM'
     &          .OR.CARD(2:7).EQ.'ENV=LU') THEN
*. Internal LUCIA environment as generated by a previous LUCIA run.
*
*. Integrals read formatted in, only integrals differing from
*. zero by symmetry are  included
          IDO_LIPKIN = 0
          INTIMP = 3
          ISETKW(24) = 1
          ENVIRO(1:6) = 'LUCIA '
          GOTO 999
        ELSE IF(CARD(2:7).EQ.'SIRIUS'.OR.CARD(2:7).EQ.'DALTON'
     &           .OR.CARD(2:7).EQ.'ENV=DA') THEN
*. Integrals imported from SIRIUS/DALTON
          IDO_LIPKIN = 0
          INTIMP = 5
C         write(6,*) ' Sirius Flag activated '
          ISETKW(24) = 1
          ENVIRO(1:6) = 'DALTON'
          GOTO 999
        ELSE IF(CARD(2:7).EQ.'LIPKIN' ) THEN
*. The Lipkin-Model
          ENVIRO(1:6) = 'LIPKIN'
c read in parameters -- preliminary values
          XLIP_V = 0.3d0
          XLIP_E = 1.0d0
          IDO_LIPKIN = 1
*. and no MO-AO file
          NOMOFL = 1
          INTIMP = 8
          ISETKW(24) = 1
          GOTO 999
        ELSE IF(CARD(2:7).EQ.'ENV=NO' ) THEN
*. No program environment, integrals, coefs will just be set to zero
          IDO_LIPKIN = 0
          ENVIRO(1:6) = 'NONE  '
          INTIMP = 0
          ISETKW(24) = 1
          GOTO 999
*. Fusk environment, integrals will be set to non-vanishing values
        ELSE IF(CARD(2:7).EQ.'ENV=FU' ) THEN
          IDO_LIPKIN = 0
          ENVIRO(1:6) = 'FUSK  '
          INTIMP = 9
          ISETKW(24) = 1
          GOTO 999
        END IF
*
* 24: Integral import
*
      IF(ISETKW(24).EQ.0) THEN
        IDO_LIPKIN = 0
        IF(IDOQD.EQ.0) THEN
*. Default is - from NOV26: Dalton
         INTIMP = 5
         ENVIRO(1:6) = 'DALTON'
        ELSE
         ENVIRO(1:6)='QDOT  '
        END IF
        ISETKW(24) = 2
      END IF
* ===============================
* 25:INCORE option for integrals
* ==============================
        IF(CARD(2:7).EQ.'INCORE') THEN
          ISETKW(25) = 1
          INCORE = 1
          GOTO 999
        END IF
* ===================
* 26: Deleted shells
* ===================
        IF(CARD(2:7).EQ.'DELETE') THEN
          ISETKW(26) = 1
          READ(LUIN,*) (NDELSH(IRREP),IRREP= 1, NIRREP)
          GOTO 999
        END IF
* ===================
* 27: Ms combinations
* ===================
        IF(CARD(2:7).EQ.'MSCOMB') THEN
          ISETKW(27) = 1
          READ(LUIN,*) PSSIGN
          IF(.NOT.(PSSIGN.EQ.-1.0D0.OR.PSSIGN.EQ.1.0D0)) THEN
            WRITE(LUOUT,*)' Illegal Spin combination factor ',PSSIGN
            ISETKW(27) = -1
            NERROR = NERROR + 1
          END IF
          GOTO 999
        END IF
* ===================
* 28: Ml combinations
* ===================
        IF(CARD(2:7).EQ.'MLCOMB') THEN
          ISETKW(28) = 1
          READ(LUIN,*) PLSIGN
          IF(.NOT.(PLSIGN.EQ.-1.0D0.OR.PLSIGN.EQ.1.0D0)) THEN
            WRITE(LUOUT,*)' Illegal ml combination factor ',PLSIGN
            NERROR = NERROR + 1
          END IF
          GOTO 999
        END IF
* ======================================
* 29: Print flag for string information
* ======================================
        IF(CARD(2:7).EQ.'IPRSTR') THEN
          ISETKW(29) = 1
          READ(LUIN,*) IPRSTR
          GOTO 999
        END IF
* ======================================
* 30: Print flag for string information
* ======================================
        IF(CARD(2:7).EQ.'IPRCIX') THEN
          ISETKW(30) = 1
          READ(LUIN,*) IPRCIX
          GOTO 999
        END IF
* ======================================
* 31: Print flag for Orbital information
* ======================================
        IF(CARD(2:7).EQ.'IPRORB') THEN
          ISETKW(31) = 1
          READ(LUIN,*) IPRORB
          GOTO 999
        END IF
* ===============================================
* 32: Print flag for diagonalization information
* ===============================================
        IF(CARD(2:7).EQ.'IPRDIA') THEN
          ISETKW(32) = 1
          READ(LUIN,*) IPRDIA
          GOTO 999
        END IF
* ===============================================
* 36: Print flag for Externals 
* ===============================================
        IF(CARD(2:6).EQ.'IPRXT') THEN
          ISETKW(36) = 1
          READ(LUIN,*) IPRXT
          GOTO 999
        END IF
* =====================================
* 43: Print occupation of lowest Dets 
* =====================================
       IF(CARD(2:7).EQ.'IPROCC') THEN
         ISETKW(43) = 1
         READ(LUIN,*) IPROCC
         GOTO 999
       END IF 
* ====================================
* 65: Print level for density matrices 
* ====================================
       IF(CARD(2:7).EQ.'IPRDEN') THEN
         ISETKW(65) = 1
         READ(LUIN,*) IPRDEN
         GOTO 999
       END IF 
* ===========================================
* 84: Print level for Response calculations
* ===========================================
       IF(CARD(2:7).EQ.'IPRRSP') THEN
         ISETKW(84) = 1
         READ(LUIN,*) IPRRSP
         GOTO 999
       END IF 
* ===========================================
* 99: Print level for Property calculations
* ===========================================
       IF(CARD(2:7).EQ.'IPRPRO') THEN
         ISETKW(99) = 1
         READ(LUIN,*) IPRPRO
         GOTO 999
       END IF 
* =======================================
* 106: Print level for CC  calculations
* =======================================
       IF(CARD(2:6).EQ.'IPRCC') THEN
         ISETKW(106) = 1
         READ(LUIN,*) IPRCC
         GOTO 999
       END IF 

*
*=========================================================
* 33: Largest allowed number of Vectors in diagonalization
*=========================================================
        IF(CARD(2:6).EQ.'MXCIV') THEN
          ISETKW(33) = 1
          READ(LUIN,*) MXCIV
          MXCIVG = MXCIV
          GOTO 999
        END IF
* =============================
* 34: Storage mode for vectors
* =============================
       IF(CARD(2:7).EQ.'CISTOR')THEN
         ISETKW(34) = 1
         READ(LUIN,*) ICISTR
         GOTO 999
       END IF
* ================================
* 35: Do not employ CSF expansion
* ================================
       IF(CARD(2:6).EQ.'NOCSF') THEN
         ISETKW(35) = 1
         NOCSF = 1
         GOTO 999
       END IF
* ================================
* 37: Do not read in integrals   
* ================================
       IF(CARD(2:6).EQ.'NOINT') THEN
         ISETKW(37) = 1
         NOINT = 1
         GOTO 999
       END IF
* ================================
* 38: Dump integrals in formatted form on file 90
* ================================
       IF(CARD(2:7).EQ.'DMPINT') THEN
         ISETKW(38) = 1
         IDMPIN = 1
         GOTO 999
       END IF
* ================================
* 39: Define dimension of resolution matrices    
* ================================
       IF(CARD(2:7).EQ.'RESDIM') THEN
         ISETKW(39) = 1
         READ(LUIN,*) MXINKA
         GOTO 999
       END IF
* ====================================================================
* 40: Use CJKAIB matrices as intermediate matrices in alpha-beta-loop
* ====================================================================
       IF(CARD(2:7).EQ.'CJKAIB') THEN
         ISETKW(40) = 1
         ICJKAIB = 1
         GOTO 999
       END IF
* ====================================================================
* 44: Use Minimal operatioon count method for alpha-alpha and beta-beta
* ====================================================================
       IF(CARD(2:6).EQ.'MOCAA') THEN
         ISETKW(44) = 1
         MOCAA = 1
         GOTO 999
       END IF
* ====================================================================
* 45: Use Minimal operatioon count method for alpha-beta               
* ====================================================================
       IF(CARD(2:6).EQ.'MOCAB') THEN
         ISETKW(45) = 1
         MOCAB = 1
         GOTO 999
       END IF
         

* ====================================================================
* 41: Initial CI in reference space                                   
* ====================================================================
       IF(CARD(2:7).EQ.'INIREF') THEN
         ISETKW(41) = 1
         INIREF  = 1
         GOTO 999
       END IF
* ====================================================================
* 42: Restart from reference CI expansion                             
* ====================================================================
       IF(CARD(2:7).EQ.'RESTRF') THEN
         ISETKW(42) = 1
         IRESTRF = 1
*. Flag that restart will be used for zero space calculation
         ISETKW(23) = 1
         IRESTR = 1
         GOTO 999
       END IF
* ====================================================================
* 46: Read in of core energy                                          
* ====================================================================
       IF(CARD(2:6).EQ.'ECORE') THEN
         ISETKW(46) = 1
         READ(LUIN,*) ECORE
         GOTO 999
       END IF
*
* =====================================================================
* 47: Use Perturbation theory for zero order space
* =====================================================================
*
       IF(CARD(2:6).EQ.'PERTU') THEN
*
*. Perturbation theory: Three parameters to be specified:
*
*      1: Max order of correction vectors required
*      2: Type of partitioning ( H0 ) 
*          Current choices: MP, EN, H0READ 
*      3: zero order energy: E0=EX ( use exact energy of reference state )
*                              E0=AV ( Use expectation value of H0 )
*                              E0=RE ; Readin zero order energy in
         ISETKW(47) = 1
         IPERT = 1
*. Number of correction vectors
         READ(LUIN,*) NPERT
*. Moeller-Plesset or Epstein-Nesbet partitioning
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
C?       WRITE(6,'(A)') CARD1
*
         IF(CARD1(1:2) .EQ. 'MP' ) THEN
           MPORENP = 1
           IPART = 1
         ELSE  IF(CARD1(1:2) .EQ. 'EN' ) THEN
           MPORENP = 2
           IPART = 2
         ELSE IF(CARD1(1:6).EQ.'H0READ' ) THEN
*. Read in one body hamiltonian
           MPORENP = 0
           IPART = 3
*.
         ELSE
           WRITE(LUOUT,*) ' Unknown partitioning '
          WRITE(LUOUT,'(1H ,A)') CARD1
          NERROR = NERROR + 1
         END IF
* Zero order energy:
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
C?       WRITE(6,'(A)') CARD1
*
         IF(CARD1(1:5).EQ.'E0=AV') THEN
           IE0AVEX = 1
         ELSE IF(CARD1(1:5).EQ.'E0=EX') THEN
           IE0AVEX = 2
         ELSE IF(CARD1(1:5).EQ.'E0=RE') THEN
           IE0AVEX = 3
           READ(LUIN,*) E0READ
           WRITE(6,*) ' Zero order energy =',E0READ
         ELSE   
           WRITE(6,*) ' Unknown form of zero order energy '
           WRITE(LUOUT,'(1H ,A)') CARD1
           NERROR = NERROR + 1
         END IF
*
         GOTO 999
       END IF

*
* =====================================================================
* 48: Approximate Hamiltonian in reference space 
* =====================================================================
*
       IF(CARD(2:7).EQ.'APRREF') THEN
         ISETKW(48) = 1
         READ(LUIN,*)  MNRS1RE,MXRS3RE
*. Moeller-Plesset or Epstein-Nesbet partitioning
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
*. Change to upper case
         CALL UPPCAS(CARD1,MXPLNC)
*
         IF(CARD1(1:2) .EQ. 'MP' ) THEN
           MPORENR = 1
         ELSE  IF(CARD1(1:2) .EQ. 'EN' ) THEN
           MPORENR = 2
         ELSE
           WRITE(LUOUT) ' Unknown partitioning '
          WRITE(LUOUT,'(1H ,A)') CARD1
          NERROR = NERROR + 1
         END IF
*
         IAPRREF = 1
         GOTO 999
       END IF
*
* =====================================================================
* 49: Approximate Hamiltonian in zero order space 
* =====================================================================
*
       IF(CARD(2:7).EQ.'APRZER') THEN
         ISETKW(49) = 1
         READ(LUIN,*)  MNRS1ZE,MXRS3ZE
*. Moeller-Plesset or Epstein-Nesbet partitioning
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
*. Change to upper case
         CALL UPPCAS(CARD1,MXPLNC)
*
         IF(CARD1(1:2) .EQ. 'MP' ) THEN
           MPORENZ = 1
         ELSE  IF(CARD1(1:2) .EQ. 'EN' ) THEN
           MPORENZ = 2
         ELSE
           WRITE(LUOUT) ' Unknown partitioning '
          WRITE(LUOUT,'(1H ,A)') CARD1
          NERROR = NERROR + 1
         END IF
         IAPRZER = 1
         GOTO 999
       END IF
*
* =====================================================================
* 50: Generalized active space concept invoked, orbital spaces
* =====================================================================
*
      IF(CARD(2:6).EQ.'GASSH') THEN
*. Generalized active space in use
        ISETKW(50) = 1
        IDOGAS = 1
        READ(LUIN,*) NGAS
        IGSFILL = 0
        DO IGAS = 1, NGAS
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
*. A line can be one of the following 
*  NIRREP numbers giving dim of each irrep for this space
* A character entry:
*                     NONE => No orbitals in this space
*                     ALL  => All remaining orbitals  in this space
*                     REST => All remaining orbitals  in this space
*. Note: Only a single space must be defined by ALL or REST
          CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          ITEMX = ITEM(1)
          IF(ITEMX(1:4).EQ.'NONE') THEN
            DO IRREP = 1, NIRREP
              NGSSH(IRREP,IGAS) = 0
            END DO
          ELSE IF(ITEMX(1:3).EQ.'ALL'.OR.ITEMX(1:4).EQ.'REST') THEN
*. Only a single space must be defined in this way
            IF(ISECFILL.NE.0.OR.IGSFILL.NE.0) THEN
              WRITE(6,*) 
     &        ' Several shell spaces defined by ALL or REST'
              WRITE(6,*)
     &        ' This confuses and upsets me '
              WRITE(6,*)
     &        '                                / Lucia '
              ISETKW(50) = -1
              NERROR = NERROR + 1
            END IF
            IGSFILL = IGAS    
          ELSE 
*. I expect that NIRREP integers are given
            IF(NITEM.NE.NIRREP) THEN
              WRITE(6,*) ' Erroneous input to GASSH: '
              WRITE(6,'(72A)') CARD1
              WRITE(6,*) ' Specify either:   NONE '
              WRITE(6,*) '                     ALL' 
              WRITE(6,*) '                    REST' 
              WRITE(6,*) ' Or NIRREP integers '
              NERROR = NERROR + 1
              ISETKW(50) = -1
            END IF
*. Well assume NIRREP integers
            DO IRREP = 1, NIRREP
              CALL CHAR_TO_INTEGER(ITEM(IRREP),NGSSH(IRREP,IGAS),
     &             MXPLNC)
            END DO
          END IF
*. Number of irreps per GAS 
C        READ(LUIN,*) (NGSSH(IRREP,IGAS),IRREP = 1, NIRREP)
        END DO
*. We will under some circumstances need the number of orbitals per subspace already 
*. in the input so calculate this - assumes again that shells = orbitals.
        DO IGAS = 1, NGAS
          NOBPT(IGAS) =  IELSUM(NGSSH(1,IGAS),NIRREP)
        END DO
        DO ISYM = 1, NIRREP
          DO IGAS = 1, NGAS
            NTOOBS(ISYM) = NTOOBS(ISYM) + NGSSH(ISYM,IGAS)
          END DO
        END DO
        GOTO 999
       END IF
*
* =====================================================================
* 51: Generalized active space occupation restrictions
* =====================================================================
*
       IF(CARD(2:7).EQ.'GASSPC') THEN
*. Orbital constraints in gas spaces
*. GASSH must have been defined before, check this
         IF(ISETKW(50).EQ.0) THEN
           WRITE(6,*) ' Dear User'
           WRITE(6,*)
           WRITE(6,*) ' GASSH must be specified before GASSPC'
           WRITE(6,*) 
     &     ' Else I do not know about the number of orbital spaces'
           WRITE(6,*) ' So I will stop '
           STOP 'READIN: put GASSH before GASSPC'
         END IF
         IDOGAS = 1
         ISETKW(51) = 1
*. Number of oribtal spaces
         READ(LUIN,*) NCISPC
         DO ISPC = 1, NCISPC
*. Two form of input pt 
*. 1: Give accumulated input for each GASpace
*. 2: Give reference plus excitation level 
*. The two possibilities are distinguished by the latter starting with R
*
       
*. Upper and lower limits for each orbital space
             IF(NGAS.NE.0) READ(LUIN,*) 
     &       (IGSOCCX(IGAS,1,ISPC),IGSOCCX(IGAS,2,ISPC),IGAS=1,NGAS)
         END DO
         GOTO 999
       END IF
       IF(CARD(2:7).EQ.'CMBSPC') THEN
*. Calculations will be performed in combination of different GAS spaces
         IDOGAS = 1
         ISETKW(52) = 1
*. Check if SEQUEN have been specified.
         IF(ISETKW(54).EQ.1) THEN
           WRITE(6,*) ' Dear user '
           WRITE(6,*)
           WRITE(6,*)' SEQUEN flag has been specified before CMBSPC'
           WRITE(6,*)' This confuses me and makes me wonder what the'
           WRITE(6,*)' meaning of everything is. '
           WRITE(6,*)' Please ensure that CMBSPC is given before SEQUEN'
           WRITE(6,*)
           WRITE(6,*)'                                  Lucia  '
           WRITE(6,*) 
           STOP'READIN: Specify CMBSPC before SEQUEN'
         END IF
*. Number of combination spaces
         READ(LUIN,*) NCMBSPC
         DO JCMBSPC  = 1, NCMBSPC
*. Number of gas spaces in this space
           READ(LUIN,*) LLCMBSPC
           LCMBSPC(JCMBSPC) = LLCMBSPC
*. Gasspaces included
           READ(LUIN,*) (ICMBSPC(IGASSPC,JCMBSPC),IGASSPC=1,LLCMBSPC)
         END DO
         GOTO 999
       END IF
*
       IF(CARD(2:7).EQ.'CICONV') THEN
*. Energy convergence of CI 
         READ(LUIN,*) THRES_E
         ISETKW(53) = 1
         GOTO 999
       END IF
*
       IF(CARD(2:7).EQ.'SEQUEN') THEN
         ISETKW(54) = 1
*
*. SEQUEN KEYWORD
*
* Form of input is
* 
* Loop over CI spaces
*  READ NCALC <= Number of calculations in this space
*  Loop Over the NCALC calculations
*    READ type_of_calculation, further info ( remember the comma)(see below)
*  End of loop over NCALC calulation
* End of loop over CI spaces
*
*. Is total number of CI spaces defined ?
         IF(ISETKW(52).EQ.0) THEN
*. Combination spaces were not explicitly defined,
*. assume each gas space is a conb space
           NCMBSPC = NCISPC
         END IF
*
         DO JCMBSPC = 1, NCMBSPC
           READ(LUIN,*) NSEQCI(JCMBSPC)
*. To avoid problem if no calculations were specified
           ICI = 0
           DO ICI = 1, NSEQCI(JCMBSPC)
*. Read in as character line, and decode
*. Format: Type of calc, further info
*. Possible types of calculations:
* =================================
*    CI: Normal  CI
*    APR-CI: CI with approximate Hamiltonian
*    PERTU : Perturbation theory, high order version with vectors on
*             disc
*    VECFREE: Various vector free calculations
*    CC    : Coupled Cluster calculation using very new routines 
*    ICCI  : Internal contracted CI   
*    ICPT  : Internal contracted    PTQ
*    ICCC  : Internal contracted CC
*    SP_MCL: Spin MCLR in the Anders-Jeppe Version
*    CC, GEN_CC: General Coupled Cluster 
*             CC => The newer codes with the correct scaling is used 
*                   (OLDCCV flag is not turned on)
*    MCSCF : MCSCF optimization 
*    NORTCI: Nonorthogonal CI 
*    NORTMC: Nonorthogonal MCSCF
*    HF    : Hartree-Fock
*    CUMULA: Generate cumulants for the wavefunction 
*             generated in this space. Should be 
*             preceeded by a wf-calculation  in this space
*    AKBKCI : AKBK-CI calculation: must follow a CI calculation
*             in a smaller space
*    AKBKMC: AKBK-MCSCF calculation: Must follow a CI calculation in 
*            a smaller space (P-space)
*
             READ(LUIN,'(A)') CARD1
             CALL LFTPOS(CARD1,MXPLNC)
             CALL UPPCAS(CARD1,MXPLNC)
             CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
*. Type of calc:
             CARDX=ITEM(1)
             CSEQCI(ICI,JCMBSPC) = ITEM(1)(1:8)
*
* CI or CI with approximate hamiltonian
*
             IF(CARDX(1:2).EQ.'CI'     .OR.
     &          CARDX(1:6).EQ.'APR-CI'     ) THEN
*. CI calculation, second item in line will be max number of its'
               IF(NITEM.EQ.1) THEN
*. No second item, use default number of iterations: maybe not 
*. defined presently, so flag by a minus and insert later
*
* At the moment: I want MAXIT as the second entry
                 WRITE(6,*) 
     &           ' ERROR:  Number_of_iterations not specified'
                 WRITE(6,*) 
     &           ' Required form of CI card is: '
                 WRITE(6,*) ' CI , Number_of_iterations'
                 ISEQCI(ICI,JCMBSPC) = -1
                 NERROR = NERROR + 1
                 ISETKW(54) = -1
               ELSE
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 I_DO_GAS = 1
               END IF
             ELSE IF(CARDX(1:5).EQ.'PERTU') THEN
               I_DO_GAS = 1
*. Perturbation calculation, following items are 
* Maxord, Ipart, E0 with
* 1) Maxord: order to which perturbation vectors will be solved 
* 2) Ipart :  Partitioning of zero order Hamiltonian,
*              MP-DIAG: Diagonal Moller-Plesset operator
*              MP-FULL: Full nondiagonal Moller-Plesset operator
*              EN     : Epstein-Nesbet: Hamiltonian diagonal
*              GENH0  : General H0, specified by separate keyword
* 3) E0    :  Definition of zero order energy
*              E0=EX: Use exact energy of zero order state 
*              E0=AV: Use average Zero order energy
*              E0=RE: Read in exact zero .
*
* First time around: No extra info, use normal perturbation keyword
* PERTU to specify perturbation calculation
*
             ELSE IF(CARDX(1:7).EQ.'VECFREE') THEN
               I_DO_GAS = 1
*
* ========================
*. Vector free calculation
* ========================
*
* Three entries: VECFREE, LEVEL, MPORENP
*
*. Second entry is level of calculation
*
*              LEVEL = 1 => second order perturbation calc
*              LEVEL = 2 => + 1 CI it + third order calc
*              LEVEL = 3 => 1 MP4 in current CI space
*              LEVEL = 4 => Level 2 + MP4 in next space
*
               IF(NITEM.LT.3) THEN
                  WRITE(6,*) 
     &            ' ERROR:  Level and MPORENP parameter not specified'
                  WRITE(6,*) 
     &            ' Required form of VECFREE card is: '
                  WRITE(6,*) ' VECFREE , LEVEL, MPORENP'
                  ISEQCI(ICI,JCMBSPC) = -1
                  NERROR = NERROR + 1
               ELSE
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
*. Level parameter is traditionally specified by negative number,
                      ISEQCI(ICI,JCMBSPC) = -ISEQCI(ICI,JCMBSPC)
                 CALL CHAR_TO_INTEGER(ITEM(3),MPORENP,MXPLNC)
C?               WRITE(6,*) ' MPORENP = ', MPORENP
               END IF
             ELSE IF(CARDX(1:2).EQ.'CC' .OR.
     &               CARDX(1:3).EQ.'TCC'.OR.
     &               CARDX(1:3).EQ.'ECC'.OR.
     &               CARDX(1:3).EQ.'VCC'.OR.
     &               CARDX(1:3).EQ.'UCC'.OR.
     &               CARDX(1:4).EQ.'URCC'.OR.
     &               CARDX(1:6).EQ.'GEN_CC' .OR.
     &               CARDX(1:3).EQ.'RCC'    ) THEN

*
* ==============================
*. Coupled Cluster calculation
* ==============================
*
C              WRITE(6,*) ' CC routines will be called '
               I_DO_GAS = 1
               I_DO_CC = 1
*. Closed shell, unrestricted or restricted form
               IF(CARDX(1:2).EQ.'CC') THEN
*, Codes with optimal scaling and HTF approach will be used.
*. The keyword NEWCCV is turned automatically on 
                 ICC_CUR = 1
               ELSE IF( CARDX(1:3).EQ.'URCC') THEN
                 ICC_CUR = 2
               ELSE IF( CARDX(1:3).EQ. 'RCC') THEN
                 ICC_CUR = 3
               ELSE IF( CARDX(1:6).EQ. 'GEN_CC'.OR.
     &                  CARDX(1:3).EQ.'TCC'.OR.
     &                  CARDX(1:3).EQ.'ECC'.OR.
     &                  CARDX(1:3).EQ.'VCC'.OR.
     &                  CARDX(1:3).EQ.'UCC'    ) THEN
*. Using either HTF approach or the first set of generalized CC codes 
* ( determined by the presence or absence of the OLDCCV keyword)
                 ICC_CUR = 4
               END IF
*. Last calculation which is CC
               LAST_CC_SPC = JCMBSPC
               LAST_CC_RUN = ICI
*
               IF(CARDX(1:2).EQ.'CC') THEN
*. The NEWCCV flag should be turned on 
COLD             ISETKW(123) = 2
                 IF(NITEM.EQ.1) THEN
* At the moment: I want MAXIT as the second entry
                   WRITE(6,*) 
     &             ' ERROR:  Number_of_iterations not specified'
                   WRITE(6,*) 
     &             ' Required form of CC card is: '
                   WRITE(6,*) ' CC , Number_of_iterations'
                   ISEQCI(ICI,JCMBSPC) = -1
                   NERROR = NERROR + 1
                   ISETKW(54) = -1
                 ELSE
                   CALL CHAR_TO_INTEGER(ITEM(2),
     &                  ISEQCI(ICI,JCMBSPC),MXPLNC)
                 END IF
               ELSE IF(CARDX(1:6).EQ.'GEN_CC'.OR.
     &                 CARDX(1:3).EQ.'TCC'.OR.
     &                 CARDX(1:3).EQ.'ECC'.OR.
     &                 CARDX(1:3).EQ.'VCC'.OR.
     &                 CARDX(1:3).EQ.'UCC'    ) THEN
* For GEN_CC The inputline should read: GEN_CC, MAXIT, ITSPC
* Where T space is CI space used to define T-operator space
               IF(NITEM.NE.3) THEN
                 WRITE(6,*) 
     &           ' ERROR:   GEN_CC card does not contain 3 items '
                 WRITE(6,*) 
     &           ' Required form of GEN_CC card is: '
                 WRITE(6,*) ' GEN_CC , MAXIT, ITSPC'
                 ISEQCI(ICI,JCMBSPC) = -1
                 NERROR = NERROR + 1
                 ISETKW(54) = -1
               ELSE
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 CALL CHAR_TO_INTEGER(ITEM(3),
     &                ISEQCI2(ICI,JCMBSPC),MXPLNC)
               END IF
               END IF
             ELSE IF(CARDX(1:4).EQ.'ICCI' ) THEN
* 
* ==============================
*. Internal contracted CI calculation
* ==============================
*
               I_DO_GAS = 1
C              WRITE(6,*) ' ICCI routines will be called '
             ELSE IF(CARDX(1:5).EQ.'GICCI') THEN
C              WRITE(6,*) ' GICCI routines will be called'
               I_DO_GIC = 1
             ELSE IF(CARDX(1:4).EQ.'ICPT' ) THEN
               I_DO_GAS = 1
*
* ======================================
*. Internal contracted PT calculation
* ======================================
*
               WRITE(6,*) ' Internal contracted PT '      
             ELSE IF(CARDX(1:4).EQ.'ICCC' ) THEN
*
* =====================================
*. Internal contracted Coupled cluster
* =====================================
*
               WRITE(6,*) ' Internal contracted CC '      
               I_DO_ICCC = 1
               I_DO_GAS = 1
             ELSE IF(CARDX(1:7).EQ.'TWOBODY' ) THEN
*
* ==============================
*. Generalized TWOBODY cluster expansion
* ==============================
*
               WRITE(6,*)
     &        ' Generalized TWOBODY cluster expansions will be tested '
               I_MODE_GTBCE = 1
               I_DO_GTBCE=1
               I_DO_GAS = 1
               ISEQCI2(ICI,JCMBSPC) = 1
               ISEQCI(ICI,JCMBSPC) = 150
               IEXTRA = 0
               DO IITEM = 2, NITEM
                 IF(ITEM(IITEM)(1:1).NE.' ') THEN
                   IF(IITEM.EQ.2)
     &                  CALL CHAR_TO_INTEGER(ITEM(2),
     &                    ISEQCI(ICI,JCMBSPC),MXPLNC)
                   IF(IITEM.EQ.3)
     &                  CALL CHAR_TO_INTEGER(ITEM(3),
     &                    ISEQCI2(ICI,JCMBSPC),MXPLNC)
                 END IF
               END DO
               
             ELSE IF(CARDX(1:6).EQ.'SP_MCL') THEN
               WRITE(6,*) ' Spin-MCLR will be called '
*
             ELSE IF(CARDX(1:6).EQ.'SP_MCL') THEN
               WRITE(6,*) ' Spin-MCLR will be called '
*
             ELSE IF (CARDX(1:5).EQ.'MCSCF') THEN
* 
* =========================
*. MCSCF Optimization                
* =========================
               I_DO_MCSCF = 1
               I_DO_GAS = 1
C?             WRITE(6,*) ' MCSCF optimization will be invoked '
* For MCSCF the inputline should read: MCSCF, MAXMAC, MAXMIC
               IF(NITEM.NE.3) THEN
                 WRITE(6,*) 
     &           ' ERROR:   MCSCF card does not contain 3 items '
                 WRITE(6,*) 
     &           ' Required form of MCSCF card is: '
                 WRITE(6,*) ' MCSCF, MAXMAC, MAXMIC'
                 ISEQCI(ICI,JCMBSPC) = -1
                 NERROR = NERROR + 1
                 ISETKW(54) = -1
               ELSE
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 CALL CHAR_TO_INTEGER(ITEM(3),
     &                ISEQCI2(ICI,JCMBSPC),MXPLNC)
               END IF
             ELSE IF (CARDX(1:6).EQ.'NORTCI') THEN
* ================================
*. Nonorthogonal CI calculations 
* ================================
               I_DO_NORTCI = 1
               I_DO_GAS = 1
*. Input line should be: NORTCI   Number_of_ci_iterations 
*                    or
*                        NORTCI   Number_of_ci_iterations VBGNSPC
               IF(NITEM.LT.2 ) THEN
                  WRITE(6,*) 
     &            ' ERROR:   NORTCI card does not contain 2 items '
                  WRITE(6,*) 
     &            ' Allowed forms of NORTCI card are: '
                  WRITE(6,*) ' NORTCI   Number_of_ci_iterations'
                  WRITE(6,*) ' NORTCI   Number_of_ci_iterations IVBGNSP'
                  ISEQCI(ICI,JCMBSPC) = -1
                  NERROR = NERROR + 1
                  ISETKW(54) = -1
               ELSE  IF (NITEM.EQ.2) THEN
*. VB calculation is in reference space
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 ISEQCI2(ICI,JCMBSPC) = 0
               ELSE IF( NITEM.GE.3) THEN
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 CALL CHAR_TO_INTEGER(ITEM(3),
     &                ISEQCI2(ICI,JCMBSPC),MXPLNC)
               END IF
             ELSE IF (CARDX(1:6).EQ.'NORTMC') THEN
* ================================
*. Nonorthogonal MCSCF calculations 
* ================================
               I_DO_NORTCI = 1
               I_DO_NORTMCSCF = 1
               I_DO_GAS = 1
*. Input line should be: NORTMC, Number of macro It, number of micro it
               IF(NITEM.NE.3) THEN
                  WRITE(6,*) 
     &            ' ERROR:   NORTMC card does not contain 3 items '
                  WRITE(6,*) 
     &            ' Required form of NORTMC card is: '
                  WRITE(6,*) 
     &            ' NORTMC, Number of macroit, Number of Microit'
                  ISEQCI(ICI,JCMBSPC) = -1
                  ISEQCI2(ICI,JCMBSPC) = -1
                  NERROR = NERROR + 1
                  ISETKW(54) = -1
               ELSE 
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 CALL CHAR_TO_INTEGER(ITEM(3),
     &                ISEQCI2(ICI,JCMBSPC),MXPLNC)
               END IF
             ELSE IF (CARDX(1:2).EQ.'HF') THEN
* ================================
*. Hartree-Fock optimization
* ================================
               I_DO_HF = 1
               IF(NITEM.NE.2) THEN
                  WRITE(6,*) 
     &            ' ERROR:   HF card does not contain 2 items '
                  WRITE(6,*) 
     &            ' Required form of HF card is: '
                  WRITE(6,*) ' HF   Number_of_hf_iterations'
                  ISEQCI(ICI,JCMBSPC) = -1
                  NERROR = NERROR + 1
                  ISETKW(54) = -1
               ELSE 
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
               END IF
             ELSE IF(CARDX(1:6).EQ.'CUMULA') THEN
* ===============================================
*. Calculate cumulants through the given order 
* ===============================================
               WRITE(6,*) ' Calculations of cumulants'
             ELSE IF (CARDX(1:6).EQ.'AKBKCI') THEN
* ===============================================
*. AKBKCI calculation
* ===============================================
               I_DO_GAS = 1
C?             WRITE(6,*) ' AKBKCI calculation '
             ELSE IF (CARDX(1:6).EQ.'AKBKMC') THEN
* ===============================================
*. AKBKMCSCF calculation
* ===============================================
               I_DO_GAS = 1
               I_DO_MCSCF = 1
               WRITE(6,*) ' AKBKMC calculation '
* For AKBKMCSCF the inputline should read: AKBKMC, MAXMAC, MAXMIC
               IF(NITEM.NE.3) THEN
                 WRITE(6,*) 
     &           ' ERROR: AKBKMC card does not contain 3 items '
                 WRITE(6,*) 
     &           ' Required form of AKBKMC card is: '
                 WRITE(6,*) ' AKBKMC, MAXMAC, MAXMIC'
                 ISEQCI(ICI,JCMBSPC) = -1
                 NERROR = NERROR + 1
                 ISETKW(54) = -1
               ELSE
                 CALL CHAR_TO_INTEGER(ITEM(2),
     &                ISEQCI(ICI,JCMBSPC),MXPLNC)
                 CALL CHAR_TO_INTEGER(ITEM(3),
     &                ISEQCI2(ICI,JCMBSPC),MXPLNC)
               END IF
C             END IF
*. No more check of input pt
             ELSE
               WRITE(6,'(A,A)') 
     &        ' Unknown type of calculation specified in SEQUEN:  ',
     &         CARDX
               WRITE(6,*) ' Allowed ENTRIES: '
               WRITE(6,*) ' ================='
               WRITE(6,*) '     CI'
               WRITE(6,*) '     APR_CI'
               WRITE(6,*) '     PERTU '
               WRITE(6,*) '     VECFREE'
               WRITE(6,*) '     CC     '
               WRITE(6,*) '     ICCI   '
               WRITE(6,*) '     GICCI  '
               WRITE(6,*) '     ICPT   '
               WRITE(6,*) '     TWOBODY'
               WRITE(6,*) '     SP_MCL '
               WRITE(6,*) '     GEN_CC '
               WRITE(6,*) '     MCSCF  '
               WRITE(6,*) '     NORTCI '
               WRITE(6,*) '     NORTMC '
               WRITE(6,*) '     HF     '
               WRITE(6,*) '     CUMULA '
               WRITE(6,*) '     AKBKCI '
               NERROR = NERROR + 1
               ISETKW(54) = -1
             END IF
           END DO
*          ^ End of loop over calculations for given CI space
         END DO
*        ^ End of loop over CI spaces
*. The old input for the SEQUEN: Short and numeric !:
C          IF(NSEQCI(JCMBSPC).GT.0)
C    &     READ(LUIN,*) (ISEQCI(ICI,JCMBSPC),ICI = 1, NSEQCI(JCMBSPC))
C        END DO
         GOTO 999
       END IF
*
* =====================================================================
* Call EXTENDED KOOPMANS' THEOREM ROUTINE
* =====================================================================
*
       IF(CARD(2:7).EQ.'EXTKOP') THEN
*. Ih yes, we will do it !
         IEXTKOP = 1
         ISETKW(55) = 1
         GOTO 999
       END IF
*
* ==========================
* 56: What's your engine ?                            
* ==========================
*
       IF(CARD(2:7).EQ.'MACHIN') THEN
         ISETKW(56) = 1
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
*. Change to upper case
         CALL UPPCAS(CARD1,MXPLNC)
         MACHINE(1:6) = CARD1(1:6)
C?       WRITE(6,'(A,A)') ' Machine = ', MACHINE
         GOTO 999
       END IF
*
* ==========================================================
* 57: Save first order correction to wavefunction on DISC?  
* ==========================================================
*
* ( For vector free calculations )
*
       IF(CARD(2:6).EQ.'C1DSC') THEN
         ISETKW(57) = 1
         IC1DSC = 1
         GOTO 999
       END IF
*
* ==========================================================
*.58:  Specify subspaces in which perturbation is nonvanishing
* ==========================================================
*
       IF(CARD(2:6).EQ.'H0SPC') THEN
*. Ensure that number of GASSPACES have been defined 
         IF(ISETKW(50).EQ.0) THEN
           WRITE(6,*) ' Dear User'
           WRITE(6,*)
           WRITE(6,*) ' GASSH must be specified before H0SPC'
           WRITE(6,*) 
     &     ' Else I do not know about the number of orbital spaces'
           WRITE(6,*) ' So I will stop '
           STOP 'READIN: put GASSH before H0SPC '
         END IF
         READ(LUIN,*) NPTSPC
         IF(NPTSPC.GT.MXPPTSPC) THEN
*
           WRITE(LUOUT,*) ' To many perturbation spaces '
           WRITE(LUOUT,*) 
     &     ' raise MXPPTSPC from ', MXPPTSPC ,' to ',NPTSPC
           STOP'NPTSPC>MXPPTSPC in READIN '
         END IF
* 
         IH0SPC = 1
         DO JPTSPC = 1, NPTSPC
*. Number of occupation spaces in this subspace
C          DO JGAS = 1, NGAS
             READ(LUIN,*)
     &       (IOCPTSPC(1,JGAS,JPTSPC),IOCPTSPC(2,JGAS,JPTSPC),
     &       JGAS = 1, NGAS)
C          END DO
         END DO
         ISETKW(58) = 2
         GOTO 999
       END IF
*
* ============================================
*.59:  Specify Type of H0 for each subspace                    
* ============================================
*
       IF(CARD(2:6).EQ.'H0FRM') THEN
*. Ensure that number of Perturbation subspaces have been defined 
         IF(ISETKW(58).EQ.0) THEN
           WRITE(6,*) ' Dear User'
           WRITE(6,*)
           WRITE(6,*) ' H0SPC must be specified before H0FRM'
           WRITE(6,*) 
     &     ' Else I do not know about the number of spaces'
           WRITE(6,*) ' So I will stop '
           STOP 'READIN: put H0SPC before H0FRM '
         END IF
*. Type of perturbation in this subspace
*
* 1 => Diagonal MP
* 2 => EN
* 3 => Nondiagonal MP
* 4 => Exact Hamiltonian 
* 5 => Nondiagonal FI+FA + exact in orbital subspaces
*
         DO JPTSPC = 1, NPTSPC
           READ(LUIN,*) IH0INSPC(JPTSPC)
         END DO
         ISETKW(59) = 2
         GOTO 999
       END IF
*
* =============================================
* 60: Reference root for Perturbation theory          
* =============================================
*
       IF(CARD(2:7).EQ.'RFROOT') THEN
         ISETKW(60) = 1
         READ(LUIN,*) IRFROOT
C        WRITE(6,*) ' Reference Root = ',IRFROOT 
         GOTO 999
       END IF
*
* ======================================================
* 61: Orbital spaces in which Exact Hamiltonian is used
* ======================================================
*
       IF(CARD(2:5).EQ.'H0EX') THEN
         ISETKW(61) = 1
         READ(LUIN,*)  NH0EXSPC
         READ(LUIN,*) (IH0EXSPC(I),I=1, NH0EXSPC)
C?       WRITE(6,*) ' Keyword: H0EX activated '
C?       WRITE(6,*) '  NH0EXSPC ',  NH0EXSPC
C?       WRITE(6,*) (IH0EXSPC(I),I=1, NH0EXSPC)
         GOTO 999
       END IF
*
* ================================================
* 62: Treatment of degenerencies of initial guess     
* ================================================
*
       IF(CARD(2:7).EQ.'INIDEG') THEN
         ISETKW(62) = 1
         READ(LUIN,*) INIDEG
         GOTO 999
       END IF
*
* ========================================================
* 63: Use modified Hamilton operator in CI optimization
* ========================================================
*
       IF(CARD(2:7).EQ.'LAMBDA') THEN
         ISETKW(63) = 1
         READ(LUIN,*) XLAMBDA
         GOTO 999
       END IF
*
* =============================================================
* 64: Length of smallest block for batch of C an Sigma vectors
* =============================================================
*
       IF(CARD(2:7).EQ.'LCSBLK') THEN
         ISETKW(64) = 1
         READ(LUIN,*) LCSBLK 
         GOTO 999
       END IF
*
*
* =============================================================
* 66: No MO-AO file
* =============================================================
*
       IF(CARD(2:7).EQ.'NOMOFL') THEN
*. No MO-AO file
         NOMOFL = 1
         ISETKW(66) = 1
         GOTO 999
       END IF
*
*
* =============================================================
* 67: ECHO the following keywords
* =============================================================
*
       IF(CARD(2:5).EQ.'ECHO') THEN
         IECHO = 1 
         ISETKW(67) = 1
         GOTO 999
       END IF
*
*
* ====================
* 68: Final orbitals              
* ====================
*
*. Should be specified after NIRREP, I have not added the 
* test!!
       IF(CARD(2:7).EQ.'FINORB') THEN
*. Type of final orbitals
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
*. Change to upper case
         CALL UPPCAS(CARD1,MXPLNC)
*
C?       WRITE(6,'(A,A)') 
C?   &   ' Type of final orbitals ',CARD1
         ISETKW(68) = 1
*
         IF(CARD1(1:5).EQ.'NATUR') THEN
*. Natural orbitals
           IFINMO = 1
         ELSE IF (CARD1(1:5).EQ.'CANON' ) THEN
*. Canonical orbitals
           IFINMO = 2
         ELSE IF ( CARD1(1:6).EQ.'PS_NAT') THEN
*. Pseudo natural orbitals
           IFINMO = 3
         ELSE IF ( CARD1(1:6) .EQ. 'PS_CAN') THEN
*. Pseudo canonical orbitals
           IFINMO = 4
          ELSE IF (CARD1(1:5) .EQ. 'PS_NC') THEN
*. Pseudo natural-canonical orbitals
           IFINMO = 5
*. requires input of subshells in which to define 
*. Pseudo-natural orbitals
           READ(LUIN,*) NPSSPC
           DO IPSSPC = 1, NPSSPC
             READ(LUIN,*) (NPSSH(IRREP,IPSSPC),IRREP=1,NIRREP)
           END DO
         ELSE
*. Unidentified type of final orbitals
           WRITE(6,*) ' Unidentified type of final orbitals'
           WRITE(6,'(A,A)') '  you suggested: ', CARD1
           WRITE(6,*) 
           WRITE(6,*) ' Allowed types of final orbitals'
           WRITE(6,*) ' ==============================='
           WRITE(6,*) 
           WRITE(6,*) '     NATUR'
           WRITE(6,*) '     CANON'
           WRITE(6,*) '     PS_NAT'
           WRITE(6,*) '     PS_CAN'
           WRITE(6,*) '     PS_NC'
           NERROR = NERROR + 1
           ISETKW(68) = - 1
         END IF
         GOTO 999
*
       END IF
*
*
* ===================================================================
* 69: Threshold on second order energy corrections, individual coefs
* ===================================================================
*
       IF(CARD(2:7).EQ.'E_THRE') THEN
         READ(LUIN,*) E_THRE
         ISETKW(69) = 1
         GOTO 999
       END IF
*
*
* =======================================================================
* 70: Threshold on first order wavefunction corrections,individual coefs 
* =======================================================================
*
       IF(CARD(2:7).EQ.'C_THRE') THEN
         READ(LUIN,*) C_THRE
         ISETKW(70) = 1
         GOTO 999
       END IF
*
* ===================================================================
* 71 Threshold on second order energy corrections, Total Threshold 
* ===================================================================
*
       IF(CARD(2:7).EQ.'E_CONV') THEN
         READ(LUIN,*) E_CONV
         ISETKW(71) = 1
         GOTO 999
       END IF
*
*
* =======================================================================
* 72: Threshold on first order wavefunction corrections,Total Threshold  
* =======================================================================
*
       IF(CARD(2:7).EQ.'C_CONV') THEN
         READ(LUIN,*) C_CONV
         ISETKW(72) = 1
         GOTO 999
       END IF
*
*
* ===============================
* 73: Selection of classes     
* ===============================
*
       IF(CARD(2:7).EQ.'CLSSEL') THEN
         ICLSSEL = 1
         ISETKW(73) = 1
         GOTO 999
       END IF
*
*
* =====================================
* 74: Calculation of density matrices 
* ======================================
*
       IF(CARD(2:6).EQ.'DENSI') THEN
         READ(LUIN,*) IDENSI
*. IDENSI = 0 => No calculation of density matrices
*  IDENSI = 1 =>  Calculation of one- body density matrix
*  IDENSI = 2 =>  Calculation of one- and two-body density matrices
         ISETKW(74) = 1
         GOTO 999
       END IF
*
*
*
* =====================================
* 75: Perturbation expansion of EKT   
* ======================================
*
       IF(CARD(2:6).EQ.'PTEKT') THEN
         IPTEKT = 1
*. Number of EKT to be analyzed, atmost 20
         READ(LUIN,*)  NPTEKT
         IF(NPTEKT.GT.20) THEN
           WRITE(6,*) ' Atmost 20 perturbation expansions'
           STOP' NPTEKT in .PTEKT to Large '
         END IF
*. orbital and symmetry for zero order solution
         DO JEKT = 1, NPTEKT
           READ(LUIN,*) LPTEKT(1,JEKT),LPTEKT(2,JEKT) 
         END DO
         ISETKW(75) = 1
C?       WRITE(6,*) ' NPTEKT = ', NPTEKT
C?       WRITE(6,*) ' LPTEKT = ',LPTEKT(1,1),LPTEKT(2,1)
         GOTO 999
       END IF
*
* =================================================
* 76: Root used to define Zero order Hamiltonian      
* =================================================
*
       IF(CARD(2:7).EQ.'H0ROOT') THEN
         ISETKW(76) = 1
         READ(LUIN,*) IH0ROOT
C        WRITE(6,*) ' Reference Root = ',IH0ROOT 
         GOTO 999
       END IF
*
* ======================================
* 77: No restart in CI calculation 2                  
* =====================================
*
       IF(CARD(2:7).EQ.'NORST2') THEN
         ISETKW(77) = 1
         IRST2 =  0
         WRITE(6,*) ' NORST2 flag read '        
         GOTO 999
       END IF
*
* =====================================================
* 78: Skip initial evaluation of energy from CI calc 2
* ====================================================
*
       IF(CARD(2:7).EQ.'SKIPEI') THEN
         ISETKW(78) = 1
         ISKIPEI =  1
         WRITE(6,*) ' SKIPEI flag set  '        
         GOTO 999
       END IF
*
* =================================================================
* 79: Symmetry of X, Y and Z - Yes it could be obtained from files
* ================================================================
*
       IF(CARD(2:7).EQ.'XYZSYM') THEN
         ISETKW(79) = 1
         READ(LUIN,*) (IXYZSYM(I),I=1,3)
C?       WRITE(6,*) 'IXYZSYM', (IXYZSYM(I),I=1,3)
         GOTO 999
       END IF
*
* ==============================================
* 80: One-electron properties to be calculated                    
* ==============================================
*
       IF(CARD(2:7).EQ.'PROPER') THEN
         ISETKW(80) = 1
         READ(LUIN,*) NPROP              
         DO IPROP = 1, NPROP
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
           PROPER(IPROP)=CARD1(1:6)
           IF(IECHO.NE.0) 
     &     WRITE(6,'(A,A)') ' Property to be calculated ',
     &     PROPER(IPROP)
         END DO
         GOTO 999
       END IF
*
* ==============================================
* 81: Transition properties                                       
* ==============================================
*
       IF(CARD(2:7).EQ.'TRAPRP') THEN
         ISETKW(81) = 1
*. Number and symmetry of additional states
         READ(LUIN,*) IEXCSYM, NEXCSTATE
C        READ(LUIN,*) NEXCSTATE
         ITRAPRP = 1
         GOTO 999
       END IF
*
* ================================
* 82: CI response calculations
* ================================
*
*. Input goes as 
*
* Labels for operators for which average values will be calculated ( A-ops)
* Number of response calculations
* Loop over calculations
* Label for pertop1, Label for pertop1, order for op1, order for op2, freq
* End of loop over calculations
* The first per operator is static, the second can be dynamic ( freq.ne.0)
* 
* Example
*
*  XDIPLEN, ZDIPLEN
*  1
*  XDIPLEN, YDIPLEN, 2, 2, 0.0D0
*  Labels of oper
       IF(CARD(2:7).EQ.'RESPON') THEN
         ISETKW(82) = 1
*. Yes I will do respons
         IRESPONS = 1
         MXNRESP =20
*. Roots on which response calculations will be carried out
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         NRSPST = NITEM
         IF(NRSPST.GT. MXNRESP) THEN
           WRITE(6,*) ' READIN: Error for keyword RESPON'
           WRITE(6,*) ' Specified number of roots = ', NRSPST
           WRITE(6,*) ' Larger than MAX = ', MXNRESP
           WRITE(6,*) ' PLEASE reduce NAVE_OP and RETURN '
           STOP ' READIN, KEYWORD RESPON: NRSPST .gt. 20 '
         END IF
         DO JITEM = 1, NITEM
           CALL CHAR_TO_INTEGER(ITEM(JITEM),IRSPRT(JITEM),MXPLNC)
         END DO
*. Labels for operators whose expectation values will be expanded
         MXNRESP =20
         DO JCHAR = 1, MXPLNC
           CARD1(JCHAR:JCHAR) = ' '
         END DO
         READ(LUIN,'(A)') CARD1
         WRITE(6,*) ' Input card with A-ops: '
         WRITE(6,'(72A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         WRITE(6,*) ' Input card with A-ops after LFTPOS + UPPCAS: '
         WRITE(6,'(72A)') CARD1
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         N_AVE_OP = NITEM
         IF(N_AVE_OP.GT. MXNRESP) THEN
           WRITE(6,*) ' READIN: Error for keyword RESPON'
           WRITE(6,*) ' Specified number of A ops = ', N_AVE_OP
           WRITE(6,*) ' Larger than MAX = ', MXNRESP
           WRITE(6,*) ' PLEASE reduce NAVE_OP and RETURN '
           STOP ' READIN, KEYWORD RESPON: NAVE_OP .gt. 20 '
         END IF
         DO JITEM = 1, NITEM
C               COP_CHARVEC(CHAR_IN,CHAR_OUT,NCHAR)
           CALL COP_CHARVEC(ITEM(JITEM),AVE_OP(JITEM),MXPLNC)
C          AVE_OP(JITEM) = ITEM(JITEM)
*. And left position 
           CALL LFTPOS(AVE_OP(JITEM),MXPLNC)
         END DO
*. Number of respons calculations to be performed
         READ(LUIN,*) NRESP
         IF(NRESP.GT. MXNRESP) THEN
           WRITE(6,*) ' READIN: Error for keyword RESPON'
           WRITE(6,*) ' Specified number of calcs = ', NRESP
           WRITE(6,*) ' Larger than MAX = ', MXNRESP
           WRITE(6,*) ' PLEASE reduce NRESP and RETURN '
           STOP ' READIN, KEYWORD RESPON: NRESP .gt. 20 '
         END IF
         DO IRESP = 1, NRESP
*. Operator1, Operator 2, Maxord for op1, Maxord for op2, freq 
* ( Remember commas in betweeen !!)
*. Read in as character line, and decode
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
           CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
*. Entries 1 and 2: the operators in character form
C               COP_CHARVEC(CHAR_IN,CHAR_OUT,NCHAR)
           CALL COP_CHARVEC(ITEM(1),RESP_OP(1,IRESP),MXPLNC)
           CALL COP_CHARVEC(ITEM(2),RESP_OP(2,IRESP),MXPLNC)
*. and left position 
           CALL LFTPOS(RESP_OP(1,IRESP),MXPLNC)
           CALL LFTPOS(RESP_OP(2,IRESP),MXPLNC)
C?         RESP_OP(1,IRESP) = ITEM(1)
C?         RESP_OP(2,IRESP) = ITEM(2)
C?         WRITE(6,'(A,A,A)') ' RESP( ,1),RESP( ,2)=  ',
C?   &     RESP_OP(1,IRESP) , RESP_OP(2,IRESP) 
*. Entries 3 and 4: integers, maxord
           CALL CHAR_TO_INTEGER(ITEM(3),MAXORD_OP(1,IRESP),MXPLNC)
           CALL CHAR_TO_INTEGER(ITEM(4),MAXORD_OP(2,IRESP),MXPLNC)
           IF(NITEM.EQ.4) THEN
*. No frequency
             RESP_W(IRESP) = 0.0
           ELSE
             CALL CHAR_TO_REAL(ITEM(5),RESP_W(IRESP),MXPLNC)
           END IF
         END DO
*
         GOTO 999
       END IF
*
* ==============================================
* 83: Max number of iterations in lin.eq
* ==============================================
*
       IF(CARD(2:7).EQ.'MXITLE') THEN
         ISETKW(83) = 1
*. Number and symmetry of additional states
         READ(LUIN,*) MXITLE               
         GOTO 999
       END IF
*
* ==============================================
* 85: Root homing                          
* ==============================================
*
       IF(CARD(2:7).EQ.'RTHOME') THEN
         ISETKW(85) = 1
         IROOTHOMING = 1
         GOTO 999
       END IF
*
* ==============================================
* 86: Allow Particle-hole simplifications 
* ==============================================
*
       IF(CARD(2:7).EQ.'USE_PH') THEN
         ISETKW(86) = 1
         IUSE_PH = 1      
         GOTO 999
       END IF
*
* ==============================================
* 87: Allow the sigma routine to take advice 
* ==============================================
*
       IF(CARD(2:7).EQ.'ADVICE') THEN
         ISETKW(87) = 1
         IADVICE = 1      
         GOTO 999
       END IF
*
* ================================================================
* 88: Transform CI vectors to alternative orbital representation
* ================================================================
*
       IF(CARD(2:6).EQ.'TRACI') THEN
         ITRACI = 1
         ISETKW(88) = 1
*. Read Form or orbitals to which expansion should be formed
*
* Two pieces of info required:
*  1: Complete rotations or just rotations internal rotations on GAS space
*      Keywords: Restrict or complete
*  2: Form of final orbitals 
*      Keywords: Canonical or Natural 
*      As usual the input is written as keyword1, keyword2
*
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         IF(NITEM.LT. 2) THEN
           WRITE(6,*) ' READIN: Error for keyword TRACI'
           WRITE(6,*) ' Number of items read ', NITEM
           WRITE(6,*) 
     &     ' Form of line should be: complete/restrict, fock/natural'
         END IF
*
         ITRACI_CR=ITEM(1)(1:8)
         ITRACI_CN=ITEM(2)(1:8)
         IF(    ITRACI_CR(1:4).NE.'REST'
     &     .AND.ITRACI_CR(1:4).NE.'COMP') THEN
           WRITE(6,*) ' Illegal entry under keyword TRACI '
           WRITE(6,*) ' Your suggestion: ', ITRACI_CR
           WRITE(6,*) ' Allowed entries: '
           WRITE(6,*) ' =================='
           WRITE(6,*)    ' COMPlete '
           WRITE(6,*)    ' RESTrict'
           NERROR = NERROR + 1
           ISETKW(88) = -1
         END IF
         IF(    ITRACI_CN(1:4).NE.'CANO'
     &     .AND.ITRACI_CN(1:4).NE.'NATU') THEN
           WRITE(6,*) ' Illegal entry under keyword TRACI '
           WRITE(6,*) ' Your suggestion: ', ITRACI_CN
           WRITE(6,*) ' Allowed entries '
           WRITE(6,*) ' =================='
           WRITE(6,*)    ' CANOnica'
           WRITE(6,*)    ' NATUral '
           NERROR = NERROR + 1
           ISETKW(88) = -1
         END IF
         GOTO 999
       END IF
*
* ====================================================
* 89: Separate strings into active and passive parts
* ====================================================
*
       IF(CARD(2:7).EQ.'USE_PA') THEN
         ISETKW(89) = 1
         IUSE_PA = 1      
         GOTO 999
       END IF
*
* ==========================================
* 90: Perturbation expansion of Fock matrix          
* ===========================================
*
       IF(CARD(2:7).EQ.'PTFOCK') THEN
         ISETKW(90) = 1
         IPTFOCK = 1      
         GOTO 999
       END IF
*
* ==============================
* 91: Print final CI vectors                         
* ==============================
*
       IF(CARD(2:7).EQ.'PRNCIV') THEN
         ISETKW(91) = 1
         IPRNCIV = 1      
         GOTO 999
       END IF
*
* =====================================================
* 92: Restart CC calculation (with coefs on LU_CCAMP)
* =====================================================
*
       IF(CARD(2:7).EQ.'RES_CC') THEN
         ISETKW(92) = 1
         I_RESTRT_CC = 1
         GOTO 999
       END IF
*
* =====================================================
* 93: End calculation with integral transformation
* =====================================================
*
       IF(CARD(2:7).EQ.'TRA_FI') THEN
         ISETKW(93) = 1
         ITRA_FI = 1
         GOTO 999
       END IF
*
* =========================================================
* 94: Initialize calculation with integral transformation
* =========================================================
*
       IF(CARD(2:7).EQ.'TRA_IN') THEN
         ISETKW(94) = 1
         ITRA_IN = 1
         GOTO 999
       END IF
*
* =========================================================
* 95: Use multispace (multigrid method )
* =========================================================
*
       IF(CARD(2:7).EQ.'MUL_SP') THEN
         ISETKW(95) = 1
         MULSPC = 1
*. First space where MULTIspace calculation is active
         READ(LUIN,*) IFMULSPC
*. Length of pattern and pattern
         READ(LUIN,*) LPAT
         READ(LUIN,*) (IPAT(I),I=1, LPAT)
         GOTO 999
       END IF
*
* =========================================================
* 96: Use Relaxed densities for properties
* =========================================================
*
       IF(CARD(2:6).EQ.'RELAX') THEN
         ISETKW(96) = 1
         IRELAX= 1
         GOTO 999
       END IF
*
* =========================================================
* 97: Expert mode: Input errors neglected 
* =========================================================
*
       IF(CARD(2:7).EQ.'EXPERT') THEN
         ISETKW(97) = 1
         IEXPERT= 1
         GOTO 999
       END IF
*
* ==================================================================================
* 98: Number of roots to be converged ( i.e. some unconverged roots may be allowed)
* ==================================================================================
*
       IF(CARD(2:7).EQ.'CNV_RT') THEN
         ISETKW(98) = 1
         READ(LUIN,*) NCNV_RT
         GOTO 999
       END IF
*
* ==================================================================================
* 100:  Use LUCIA for QDOT calculation
* ==================================================================================
*
       IF(CARD(2:6) .EQ. 'QDOTS') THEN
         IDOQD = 1
         ISETKW(100) = 1
*. Set environment to QDOTS
         ENVIRO(1:4) = 'QDOT'
         GOTO 999
       END IF
*
* ==================================================================================
* 101:  Restrict Ms2 at intermediate gaslevel
* ==================================================================================
*
       IF(CARD(2:7) .EQ. 'RE_MS2') THEN
         READ(LUIN,*) I_RE_MS2_SPACE,I_RE_MS2_VALUE
         ISETKW(101) = 1
         GOTO 999
       END IF
*
* ==================================================================================
* 102:  Preconditioner
* ==================================================================================
*
       IF(CARD(2:7) .EQ. 'PRECON') THEN
         ISETKW(102) = 1
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         CARDX = ITEM(1)
         IF(CARDX(1:6).EQ.'SD-DIA') THEN 
           IPRECOND = 1
         ELSE IF( CARDX(1:6).EQ.'CN-DIA') THEN
           IPRECOND = 2
         ELSE
           WRITE(6,*) ' Illegal form of preconditioner:'  
           WRITE(6,'(A,A)') ' Your suggestion: ', CARDX     
           WRITE(6,*) ' Allowed entries: '
           WRITE(6,*) ' =================='
           WRITE(6,*)    'SD-DIA'
           WRITE(6,*)    'CN-DIA'
           NERROR = NERROR + 1
           ISETKW(102) = -1
         END IF
         GOTO 999
       END IF
*
* ==================================================================================
* 103:  Treat all symmetryblocks with given type simultaneously
* ==================================================================================
*
       IF(CARD(2:7) .EQ. 'SIMSYM') THEN
         ISIMSYM = 1
         ISETKW(103) = 1
         GOTO 999
       END IF
*
* ==================================================================================
* 104  Use hardwired routines for certain sigma terms  
* ==================================================================================
*
       IF(CARD(2:7) .EQ. 'USE_HW') THEN
         IUSE_HW = 1
         ISETKW(104) = 1
         GOTO 999
       END IF
*
* ==================================================================================
* 105:  Use Full H0 including projection operators in Lambda caLCULATIONS
* ==================================================================================
*
       IF(CARD(2:7) .EQ. 'USEH0P') THEN
         IUSEH0P = 1
         ISETKW(105) = 1
         GOTO 999
       END IF
*
* ==========================================
* 107: Calculate expectation value of Lz^2
* ==========================================
       IF(CARD(2:4).EQ.'LZ2') THEN
         ISETKW(107) = 1
         I_DO_LZ2 = 1
         GOTO 999
       END IF 
*
* ==========================================
* 108: Method used for solving CC equations
* ==========================================
       IF(CARD(2:7).EQ.'CCSOLV') THEN
*
c set default method: DIIS with 8 vectors
c variational method? we assume no (ivar=0), has to be corrected later
         ivar = 0               
         iorder = 1
         iprecnd = 1
         isubsp = 2
         ilsrch = 0
         icnjgrd = 0
         mxsp_sbspja = 0
         isbspjatyp = 0
         isbspja_start = 2      ! lowest possible iteration is 2
         thr_sbspja = 1d-1
         mxsp_diis = 8
         idiistyp = 2
         idiis_start = 0
         thr_diis = 1d-1
c trust radius: not active
         trini = 1.5d0
         trmin = 0.25d0
         trmax = 2.0d0
         trthr1l = 0.8d0
         trthr1u = 1.2d0
         trthrfac1 = 1.2d0
         trthr2l = 0.3d0
         trthr2u = 2.0d0
         trfac1  = 1.2d0
         trfac2  = 0.8d0
         trfac3  = 0.3d0


         ISETKW(108) = 1
         I_SUB_KW = 1
c scan for sub-keywords
         DO WHILE(I_SUB_KW.EQ.1)
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
           CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
           IF (NITEM.EQ.0) CYCLE
           CARDX = ITEM(1)
c set the basic methods:
c    PERT: simple perturbation step
c    ASSJ: approximate subspace jacobian
c    DIIS: DIIS; can also be specified together with ASSJ (but does
c          not really improve anything then)
           IF(CARDX(1:4).EQ.'PERT') THEN 
c for old version:
             ICCSOLVE = 1
c for new version:
c unset the diis
             isubsp = 0
           ELSE IF( CARDX(1:4).EQ.'DIIS') THEN
c syntax:
c DIIS,<maxvec>,<startit>,<thresh>,<type>
             DO IITEM=2,NITEM
* dimension of DIIS space given?
               IF (ITEM(IITEM)(1:1).NE.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_INTEGER(ITEM(2),mxsp_diis,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_INTEGER(ITEM(3),idiis_start,MXPLNC)
                 IF (IITEM.EQ.4)
     &                CALL CHAR_TO_REAL(ITEM(4),thr_diis,MXPLNC)
                 IF (IITEM.EQ.5)
     &                CALL CHAR_TO_INTEGER(ITEM(5),idiistyp,MXPLNC)
               END IF
             END DO
c             IF (mxsp_diis.LT.1) THEN
c               WRITE(6,*) 'WARNIG: DIIS space out of bounds (',
c     &                    mxsp_diis,')'
c               ISETKW(108) = -1
c             END IF
c for old version:
             ICCSOLVE = 2
             MAX_DIIS_VEC=mxsp_diis
           ELSE IF( CARDX(1:4).EQ.'ASSJ') THEN
c syntax:
c ASSJ,<maxvec>,<startit>,<thresh>,<type>
c unset DIIS
             isubsp = 0
c set defaults:
             iprecnd = 2
             mxsp_sbspja = 8
             isbspjatyp = 1
             isbspja_start = 2
             thr_sbspja = 1d-1
             
             DO IITEM = 2, NITEM
               IF(ITEM(IITEM)(1:1).ne.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_INTEGER(ITEM(2),mxsp_sbspja,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_INTEGER(ITEM(3),isbspja_start,MXPLNC)
                 IF (IITEM.EQ.4)
     &                CALL CHAR_TO_REAL(ITEM(4),thr_sbspja,MXPLNC)
                 IF (IITEM.EQ.5)
     &                CALL CHAR_TO_INTEGER(ITEM(5),isbspjatyp,MXPLNC)
               END IF
             END DO
c old version:
             ICCSOLVE = 1
             I_DO_SBSPJA = 1
c max subspace dimension -- unused for the moment
             MAX_VEC_APRJ = mxsp_sbspja
c max steplength
             XMXSTP=0.22d0
c max steplength to begin with sampling of subspace
             XMXSTP_APRJ=0.22d0
           ELSE IF( CARDX(1:6).EQ.'NEWTON') THEN             
             ! modify defaults for second-order solver
             iorder = 2
             iprecnd = 1
             isubsp = 0
             trini = 0.5d0
             trmin = 0.05d0
             trmax = 1.5d0
c more detailed settings for experts:
c  ORDER
c  PRCND
c  SUBSP
c  LNSRCH
c  CONJGR
c  _DIIS
c  _SBSPJ
c  TRUSTR
c  MICIMX
           ELSE IF( CARDX(1:5).EQ.'ORDER') THEN
c ORDER of method (1st-order, 2nd-order)
             IF(NITEM.GT.1) THEN
               CALL CHAR_TO_INTEGER(ITEM(2),iorder,MXPLNC)
             END IF
           ELSE IF( CARDX(1:5).EQ.'PRCND') THEN
c type of preconditioning method (0: no, 1: diag, 2: subspace)
             IF(NITEM.GT.1) THEN
               CALL CHAR_TO_INTEGER(ITEM(2),iprecnd,MXPLNC)
             END IF
           ELSE IF( CARDX(1:5).EQ.'SUBSP') THEN
c type of subspace exploitation (0: no, 1: conj. grad.s, 2: DIIS extrap.)
             IF(NITEM.GT.1) THEN
               CALL CHAR_TO_INTEGER(ITEM(2),isubsp,MXPLNC)
             END IF
           ELSE IF( CARDX(1:6).EQ.'LNSRCH') THEN
c type of linesearch (0: no, 1: one-point est., 2: two-point est.)
             IF(NITEM.GT.1) THEN
               CALL CHAR_TO_INTEGER(ITEM(2),ilsrch,MXPLNC)
             END IF
           ELSE IF( CARDX(1:6).EQ.'CONJGR') THEN
c conj. grad. method (1: orth., 2: Polack-Ribiere, 3: Fletcher-Reeves)
             IF(NITEM.GT.1) THEN
               isubsp=1
               CALL CHAR_TO_INTEGER(ITEM(2),icnjgrd,MXPLNC)
             END IF
           ELSE IF( CARDX(1:5).EQ.'_DIIS') THEN
c expert version without changing other defaults
c syntax:
c DIIS,<maxvec>,<startit>,<thresh>,<type>
             DO IITEM=2,NITEM
* dimension of DIIS space given?
               IF (ITEM(IITEM)(1:1).NE.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_INTEGER(ITEM(2),mxsp_diis,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_INTEGER(ITEM(3),idiis_start,MXPLNC)
                 IF (IITEM.EQ.4)
     &                CALL CHAR_TO_REAL(ITEM(4),thr_diis,MXPLNC)
                 IF (IITEM.EQ.5)
     &                CALL CHAR_TO_INTEGER(ITEM(5),idiistyp,MXPLNC)
               END IF
             END DO
           ELSE IF( CARDX(1:5).EQ.'_ASSJ') THEN
c expert version without changing other defaults
c syntax:
c ASSJ,<maxvec>,<startit>,<thresh>,<type>
c set defaults: (but only those controlled by this keyword!)
             mxsp_sbspja = 8
             isbspjatyp = 1
             isbspja_start = 2
             thr_sbspja = 1d-1
             
             DO IITEM = 2, NITEM
               IF(ITEM(IITEM)(1:1).ne.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_INTEGER(ITEM(2),mxsp_sbspja,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_INTEGER(ITEM(3),isbspja_start,MXPLNC)
                 IF (IITEM.EQ.4)
     &                CALL CHAR_TO_REAL(ITEM(4),thr_sbspja,MXPLNC)
                 IF (IITEM.EQ.5)
     &                CALL CHAR_TO_INTEGER(ITEM(5),isbspjatyp,MXPLNC)
               END IF
             END DO
           ELSE IF( CARDX(1:6).EQ.'TRUSTR') THEN
c syntax:
c TRUSTR,<tr_ini>,<tr_min>,<tr_max>,<good_fac>
c                                   <thr1_upper>,<thr1_lower>,<bad_fac1>,
c                                   <thr2_upper>,<thr2_lower>,<bad_fac2>
             DO IITEM=2,NITEM
* dimension of DIIS space given?
               IF (ITEM(IITEM)(1:1).NE.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_REAL(ITEM(2),trini,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_REAL(ITEM(3),trmin,MXPLNC)
                 IF (IITEM.EQ.4)
     &                CALL CHAR_TO_REAL(ITEM(4),trmax,MXPLNC)
                 IF (IITEM.EQ.5)
     &                CALL CHAR_TO_REAL(ITEM(5),trfac1,MXPLNC)
                 IF (IITEM.EQ.6)
     &                CALL CHAR_TO_REAL(ITEM(6),trthr1u,MXPLNC)
                 IF (IITEM.EQ.7)
     &                CALL CHAR_TO_REAL(ITEM(7),trthr1l,MXPLNC)
                 IF (IITEM.EQ.8)
     &                CALL CHAR_TO_REAL(ITEM(8),trfac2,MXPLNC)
                 IF (IITEM.EQ.9)
     &                CALL CHAR_TO_REAL(ITEM(9),trthr2u,MXPLNC)
                 IF (IITEM.EQ.10)
     &                CALL CHAR_TO_REAL(ITEM(10),trthr2l,MXPLNC)
                 IF (IITEM.EQ.11)
     &                CALL CHAR_TO_REAL(ITEM(11),trfac3,MXPLNC)
               END IF
             END DO
             IF (TRMIN.GT.TRMAX.OR.
     &           TRINI.LT.TRMIN.OR.
     &           TRINI.GT.TRMAX) THEN
               WRITE(6,'(/X,A,3(/,X,A,E10.4),/)')
     &              'ERROR: Inconsistent input for trust radius: ',
     &              '  min. value  = ',TRMIN,
     &              '  init. value = ',TRINI,
     &              '  max. value  = ',TRMAX
               NERROR = NERROR + 1
               ISETKW(108) = -1
             END IF
           ELSE IF( CARDX(1:4).EQ.'MICIMX') THEN
c syntax:
c MICIMX,<micifac>,<micimac>
* total          max. number of micro-iterations: micifac1*maxiter
             micifac = 30
* per macroiter. max. number of micro-iterations: micimac
             micimac = 60

             DO IITEM=2,NITEM
               IF (ITEM(IITEM)(1:1).NE.' ') THEN
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_INTEGER(ITEM(2),micifac,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_INTEGER(ITEM(3),micimac,MXPLNC)
               END IF
             END DO
           ELSE
             ! oops, put the keyword back and let the others look
             ! at it
             I_SUB_KW = 0
             BACKSPACE(LUIN)
           END IF
         END DO
         GOTO 999
       END IF 
* ==========================================
* 109: Calculate approximate CCN Jacobian 
* ==========================================
       IF(CARD(2:4).EQ.'CCN') THEN
         ISETKW(109) = 1
         I_DO_CCN = 1
         GOTO 999
       END IF 
* ==============================================================
* 110: Calculate Jacobian in subspace to improve CC convergence 
* ==============================================================
c deactivated - deactivation deactivated, Oct.09
        IF(CARD(2:7).EQ.'SBSPJA') THEN
          ISETKW(110) = 1
          I_DO_SBSPJA = 1
          MAX_VEC_APRJ = 8
          GOTO 999
        END IF 
* =====================================================================
* 111: Convergence threshold for Coupled cluster method, norm 
* of residual.
* =====================================================================
       IF(CARD(2:7).EQ.'CCCONV') THEN
c syntax: <thrgrd>,<thr_de>,<thr_stp>         
         ISETKW(111) = 1
c defaults         
         thrstp  = 1d-6
         thrgrd  = 1d-6
         thr_de  = 1d-7

         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         
         DO IITEM = 1, NITEM
               IF(ITEM(IITEM)(1:1).ne.' ') THEN
                 IF (IITEM.EQ.1)
     &                CALL CHAR_TO_REAL(ITEM(1),thrgrd,MXPLNC)
                 IF (IITEM.EQ.2)
     &                CALL CHAR_TO_REAL(ITEM(2),thr_de,MXPLNC)
                 IF (IITEM.EQ.3)
     &                CALL CHAR_TO_REAL(ITEM(3),thrstp,MXPLNC)
               END IF           
         END DO

         GOTO 999
       END IF 
*
* ==============================================================================
* 112:  Number of gasorbital spaces that corresponds to hole spaces
* ==============================================================================
*
* ( for hole-electron separation in QDOT calculations, 
*   not for particle-hole simplifations in sigma calcs  
       IF(CARD(2:7) .EQ. 'NHOSPC') THEN
         READ(LUIN,*) N_HOLE_ORBSPACE
         ISETKW(112) = 1
         GOTO 999
       END IF
* ==========================================
* 113: Specific CC3 implementation        
* ==========================================
       IF(CARD(2:4).EQ.'CC3') THEN
         ISETKW(113) = 1
         I_DO_CC3 = 1
         GOTO 999
       END IF 
* ==================================================
* 114: Start CC with reformation of CI coefficients
* ==================================================
       IF(CARD(2:7).EQ.'CI=>CC') THEN
         ISETKW(114) = 1
         I_DO_CI_TO_CC = 1
         GOTO 999
       END IF 
* ==================================================
* 115: Form of coupled cluster expansion
* ==================================================
       IF(CARD(2:7).EQ.'CCFORM') THEN
         WRITE(6,*) 'WARNING: Keyword CCFORM is obsolete'
         WRITE(6,*) ' specify the appropriate CC-variant under SEQUEN'
         ISETKW(115) = 1
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
         CARDX = ITEM(1)
         IF(CARDX(1:3).EQ.'TCC') THEN 
           CCFORM(1:3)  = 'TCC' 
         ELSE IF( CARDX(1:3).EQ.'VCC') THEN
           CCFORM(1:3)  = 'VCC' 
         ELSE
           WRITE(6,*) ' Illegal form of CCFORM '  
           WRITE(6,'(A,A)') ' Your suggestion: ', CARDX     
           WRITE(6,*) ' Allowed entries: '
           WRITE(6,*) ' =================='
           WRITE(6,*)    'TCC' 
           WRITE(6,*)    'VCC' 
           NERROR = NERROR + 1
           ISETKW(115) = -1
         END IF
         GOTO 999
       END IF 
*
* =======================================
* 116: Calculate CC excitation energies               
* =======================================
       IF(CARD(2:7).EQ.'CCEX_E') THEN
         ISETKW(116) = 1
         I_DO_CC_EXC_E = 1
*. NIRREP must hav been defined
         IF(ISETKW(3).EQ.0) THEN
           WRITE(6,*) ' ERROR: CCEX_E must be specified after NIRREP'
           NERROR = NERROR + 1
           ISETKW(116) = -1
         ELSE
           I_SUB_KW = 1
c defaults:
           NEXC_PER_SYM(1) = 1
           NEXC_PER_SYM(2:NIRREP) = 0
           CCEX_CONV = 1D-6
           ICCEX_SLEQ = 0       ! default is set later
c scan for sub-keywords
           DO WHILE(I_SUB_KW.EQ.1)
             READ(LUIN,'(A)') CARD1
             CALL LFTPOS(CARD1,MXPLNC)
             CALL UPPCAS(CARD1,MXPLNC)
             CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
             IF (NITEM.EQ.0) CYCLE
             CARDX = ITEM(1)
             IF (CARDX(1:6).EQ.'NROOTS') THEN
               DO IITEM = 2, MIN(NITEM,NIRREP+1)
                 IF (ITEM(IITEM)(1:1).NE.' ') THEN
                   CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &                  NEXC_PER_SYM(IITEM-1),MXPLNC)
                 END IF
               END DO
             ELSE IF (CARDX(1:4).EQ.'CONV') THEN
               IF (ITEM(2)(1:1).NE.' ') THEN
                 CALL CHAR_TO_REAL(ITEM(2),
     &                CCEX_CONV,MXPLNC)
               END IF
             ELSE IF (CARDX(1:3).EQ.'REQ') THEN
               ICCEX_SLEQ = IOR(ICCEX_SLEQ,2)
             ELSE IF (CARDX(1:3).EQ.'LEQ') THEN
               ICCEX_SLEQ = IOR(ICCEX_SLEQ,1)
             ELSE
             ! put the keyword back and let the others look at it
               I_SUB_KW = 0
               BACKSPACE(LUIN)
             END IF
           END DO
           ! if nothing was specified: solve right equations:
           IF(ICCEX_SLEQ.EQ.0) ICCEX_SLEQ = 2
         END IF
         GOTO 999
       END IF 
*
* 117: Restart first calculation of ccexcitation operators 
*       
       IF(CARD(2:7).EQ.'RES_EX') THEN
         ISETKW(117) = 1
         IRES_EXC = 1          
         GOTO 999
       END IF 
*
* 118: Dimension of resolution strings for CC
*
       IF(CARD(2:7).EQ.'RESDCC') THEN
         ISETKW(118) = 1
         READ(LUIN,*) MXINKA_CC           
         LCCB = MXINKA_CC
         GOTO 999
       END IF 
*
* 119: Use Combinations for CC expansion 
*
       IF(CARD(2:7).EQ.'CMB_CC') THEN
         MSCOMB_CC = 1
         ISETKW(119) = 1
         GOTO 999
       END IF
*
* 120: Use similarity transformed Hamiltonian for singles 
*
      IF(CARD(2:7).EQ.'SIMTRH') THEN
        ISIMTRH = 1
        ISETKW(120) = 1
        GOTO 999
      END IF
*
* 121: Freeze certain excitation levels in CC calculation
*
      IF(CARD(2:7).EQ.'FRZ_CC') THEN
        IFRZ_CC = 1
        READ(LUIN,*) NFRZ_CC
        READ(LUIN,*) (IFRZ_CC_AR(I),I=1, NFRZ_CC)
        ISETKW(121) = 1
        GOTO 999
      END IF
*
* 122: Obtain CC expectaion value of energy
*
      IF(CARD(2:7).EQ.'CC_EXP') THEN
        I_DO_CC_EXP = 1
        ISETKW(122) = 1
        GOTO 999
      END IF
*
* 123: Use old CC vector function routine 
*
      IF(CARD(2:7).EQ.'OLDCCV') THEN
        I_DO_NEWCCV = 0
        ISETKW(123) = 1
        GOTO 999
      END IF
*
* 124: Use new phase convention for CC operators 
*       (O(ca)O(cb)O(aa)O(Oab), where all strings have occupations 
*        in ascending order
*
      IF(CARD(2:7).EQ.'NEWCCP') THEN
        I_USE_NEWCCP = 1
        ISETKW(124) = 1
        GOTO 999
      END IF
*
* 125: Impose a largest allowed excitation level 
*       for the spinorbital excitatations
      IF(CARD(2:7).EQ.'MXSPOX') THEN
        READ(LUIN,*) MXSPOX 
        ISETKW(125) = 1
        GOTO 999
      END IF
*        
* 126: Define a mask SD to define hole and annihilation spinorbitals
*       (used for CAS states)
      IF(CARD(2:7).EQ.'MASKSD') THEN
        I_DO_MASK_CC = 1
        DO IAB = 1, 2
*. line with occupied alpha-electrons
         READ(LUIN,'(A)') CARD1
C             DECODE2_LINE(LINE,NCHAR,NENTRY,IENTRY,MXENTRY)
         CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
C CHAR_TO_INTEGER(CHAR_X,INT_X,L_CHAR_X)
         IF(IAB.EQ.1) THEN
          MSK_AEL = NITEM
         ELSE 
          MSK_BEL = NITEM
         END IF
         DO I = 1, NITEM
           CALL CHAR_TO_INTEGER(ITEM(I),MASK_SD(I,IAB),MXPLNC)
         END DO
        END DO
        ISETKW(126) = 1
        WRITE(6,*) ' I_DO_MASK_CC (1) = ', I_DO_MASK_CC
        GOTO 999
      END IF
*
* 127: No only active rotations 
*
      IF(CARD(2:7).EQ.'NOAAEX') THEN
        NOAAEX = 1
        ISETKW(127) = 1
        GOTO 999
      END IF
*
* 128: Spin restricted coupled cluster/perturbation theory
*
      IF(CARD(2:7).EQ.'SPINRS') THEN
        ISPIN_RESTRICTED = 1
        ISETKW(128) = 1
        GOTO 999
      END IF
*
* 129: Calculate very general density matrix 
*
*
      IF(CARD(2:7).EQ.'GENTRD') THEN
*. Calculate very general transition density matrix
*
       IGENTRD = 1
*. Symmetry and MS2 of general state for transition density 
       READ(LUIN,*) IGST_SM,IGST_MS2
*. Occupation of general state for transition density 
*. Should contain fewer particles than reference state!
       READ(LUIN,*) ((IGST_OCC(ISPC,IMINMAX),IMINMAX=1,2),ISPC=1,NGAS)
       ISETKW(129) = 1
       GOTO 999
      END IF
*
* 130: Reorder some orbitals before calculations 
*
      IF(CARD(2:7).EQ.'REO_OR') THEN
*. Orbitals will be reordered compared to input 
        I_DO_REO_ORB = 1
*. Reordering is defined as a number of orbital switches
*. Each switch is defined by a symmetry, old number, new number 
*. THe orbital numbers are defined as relative to start of 
*. given orbital symmetry
        READ(LUIN,*) NSWITCH
        WRITE(6,*)  ' NSWITCH = ', NSWITCH   
        DO ISWITCH = 1, NSWITCH
          READ(LUIN,*) IREO_ORB(1,ISWITCH), IREO_ORB(2,ISWITCH),
     &                 IREO_ORB(3,ISWITCH)
        END DO
        ISETKW(130) = 1
       GOTO 999
      END IF
*
* 131: Data for internal contraction
*
      IF(CARD(2:7).EQ.'IC_EXC') THEN
*. Min and max rank of orbital excitations, are internal excitations 
*. allowed ?, and max number of active indeces (added oct 2006) 
*.
*. Change of March 2009: Old form eliminated and ICEXC_MAX_SEC added
*
        READ(LUIN,*) ICOP_RANK_MAX,
     &               ICEXC_MAX_ACT,ICEXC_MAX_EXT, ICEXC_INT
        WRITE(6,*) 
     &  'Readin: ICOP_RANK_MAX, ICEXC_MAX_ACT,ICEXC_MAX_EXT, ICEXC_INT',
     &           ICOP_RANK_MAX, ICEXC_MAX_ACT,ICEXC_MAX_EXT, ICEXC_INT
* ICOP_RANK_MAX: Max number of creation or annihilations in operator
* ICEXC_MAX_ACT: Max number of active (internal) indeces
* ICEXC_MAX_EXT: Max number of external (sec + inac) indeces
* ICEXC_INT = 0 => no creations in inactive or annihilations in secondary
* 
* Whether or not pure active excitations are allowed is flagged 
* by keyword INC_AA (default is no way...)
* 
        ISETKW(131) = 1
        I_HAVE_ICEXC_INFO = 1
        GOTO 999
      END IF
*
* 132: Expand final CC vector in last CC calculation and compare with 
*       vector on LU17
*
      IF(CARD(2:7).EQ.'CMPCCI') THEN
        ISETKW(132) = 1
        I_DO_CMPCCI = 1
        GOTO 999
      END IF
*. 133: Enforce program to rewrite CC expansion as CI expansion after last 
*. CC calc. This expansion will always be done if there is a 
*. CI calculation following the last CC calculation
*  
      IF(CARD(2:7).EQ.'CC=>CI') THEN
        I_DO_CC_TO_CI = 1
        ISETKW(133) = 1
        GOTO 999
      END IF
*
*. 134: Construct complete Hamiltonian 
*
      IF(CARD(2:7).EQ.'COMHAM') THEN
        I_DO_COMHAM = 1
        ISETKW(134) = 1
        GOTO 999
       END IF
*
*. 135: Dump Complete Hamiltonian for initial MRPT program of Lasse
*
       IF(CARD(2:7).EQ.'DMPMRP') THEN
        I_DO_DUMP_FOR_MRPT = 1
        ISETKW(135) = 1
        GOTO 999
       END IF
*
*. 136: Use the very new CC codes, started in 2001 - and completed in ???
*
      IF(CARD(2:7).EQ.'VNEWCC') THEN
       I_DO_NEWCCV = 2
       ISETKW(136) = 1
       GOTO 999
      END IF
*
*. 137: Use the very old CC codes - with erroneous scaling 
*     dublicate therefore deactivated
*
c      IF(CARD(2:6).EQ.'OLDCCV') THEN
c       I_DO_NEWCCV = 0
c       ISETKW(137) = 1
c       GOTO 999
c      END IF
*
*. 138: Generation of initial orbitals
      IF(CARD(2:7).EQ.'HF_INI') THEN
         READ(LUIN,'(A)') CARD1
         CALL LFTPOS(CARD1,MXPLNC)
         CALL UPPCAS(CARD1,MXPLNC)
         ISETKW(138) = 1
         IF(CARD1(1:5).EQ.'H1DIA') THEN 
           INI_HF_MO = 1
         ELSE IF(CARD1(1:6).EQ.'READIN') THEN
           INI_HF_MO = 2
         ELSE 
           WRITE(6,*) ' Illegal form of HF_INI :'  
           INI_HF_MO = 0
           WRITE(6,'(A,A)') ' Your suggestion: ', CARD1     
           WRITE(6,*) ' Allowed entries: '
           WRITE(6,*) ' =================='
           WRITE(6,*)    'H1DIA' 
           WRITE(6,*)    'READIN' 
           NERROR = NERROR + 1
           ISETKW(138) = -1
         END IF
         GOTO 999
      END IF
      IF(CARD(2:7).EQ.'HFSOLV') THEN
        READ(LUIN,'(A)') CARD1
        CALL LFTPOS(CARD1,MXPLNC)
        CALL UPPCAS(CARD1,MXPLNC)
        ISETKW(139) = 1
*
        IF(CARD1(1:3).EQ.'R-H') THEN
*. Roothaan-Hall procedure
           IHFSOLVE = 1
        ELSE IF (CARD1(1:4).EQ.'EOPD') THEN
           IHFSOLVE = 2
        ELSE IF (CARD1(1:6).EQ.'ONE-ST') THEN
           IHFSOLVE = 3
        ELSE IF (CARD1(1:2).EQ.'NR'.OR.CARD1(1:2).EQ.'QC') THEN
           IHFSOLVE = 4
        ELSE
          WRITE(6,*) ' Illegal form of HFSOLV :'  
          INI_HF_MO = 0
          WRITE(6,'(A,A)') ' Your suggestion: ', CARD1     
          WRITE(6,*) ' Allowed entries: '
          WRITE(6,*) ' =================='
          WRITE(6,*)    'R-H' 
          WRITE(6,*)    'EOPD' 
          WRITE(6,*)    'ONE-ST' 
          WRITE(6,*)    'QC',' or ' ,'NR' 
          NERROR = NERROR + 1
          ISETKW(139) = -1
        END IF
        GOTO 999
      END IF
*
*. Calculate one- and two- particle spin density matrices 
*
      IF(CARD(2:7).EQ.'SPNDEN') THEN
*. Read integer telling whether one- or one- and two-particle-spindensity 
*  should be calculated
        READ(LUIN,*) ISPNDEN
* ISPNDEN = 0 => do not calculate any spindensities 
* ISPNDEN = 1 => calculate one-particle spin-density
* ISPNDEN = 2 => calculate one- and two-particle spindensity
        IF(ISPNDEN.NE.0.AND.ISPNDEN.NE.1.AND.ISPNDEN.NE.2) THEN
          WRITE(6,*) ' Illegal value following KEYWORD SPNDEN '
          WRITE(6,*) ' ======================================='
          WRITE(6,*)
          WRITE(6,*) ' Allowed values are 0,1,2 '
          WRITE(6,*) ' Given value ', ISPNDEN
          ISETKW(140) = -1
        END IF
        ISETKW(140) = 1
        GOTO 999
      END IF
*. 141: specification of the general twobody operators
      IF(CARD(2:7).EQ.'GTBOPS'.OR.
     &   CARD(2:7).EQ.'GTBOPT'   ) THEN
        ISETKW(141) = 1
        IGTBMOD=0
        IGTB_DISPTT=0
        IGTB_CLOSED=0
        IGTB_TEST_H1=0
        IGTBCS= 0
        IGTB_PRJOUT=0
        ISYMMET_G = 0
        INC_SING(1:3) = 0
        INC_DOUB(1:5) = 0
        I_DO_H0=1
        EXPG_THRSH = 1D-20
        MXTERM_EXPG=200
c scan for sub-keywords
        I_SUB_KW = 1
        DO WHILE(I_SUB_KW.EQ.1)
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
          CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          IF (NITEM.EQ.0) CYCLE
c specify single-excitation operators
c SING,<n(rank +1)>,<n(rank 0)>,<n(rank -1)>
          IF (ITEM(1)(1:4).EQ.'SING') THEN
            DO IITEM = 2, NITEM
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.LE.4) THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &               INC_SING(IITEM-1),MXPLNC)
              END IF
            END DO
c specify single-excitation operators
c DOUB,<n(rank +2)>,<n(rank +1)>,<n(rank 0)>,<n(rank -1)>,<n(rank -2)>
          ELSE IF (ITEM(1)(1:4).EQ.'DOUB') THEN
            DO IITEM = 2, NITEM
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.LE.6) THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &               INC_DOUB(IITEM-1),MXPLNC)
              END IF
            END DO
c specify Hermitian symmetry of operator
          ELSE IF (ITEM(1)(1:6).EQ.'HERMIT') THEN
            IGTBCS=+1
c specify Anti-Hermitian symmetry of operator (exp(G) is unitary then)
          ELSE IF (ITEM(1)(1:6).EQ.'UNITAR') THEN
            IGTBCS=-1
c specify unsymmetric operator (default anyway)
          ELSE IF (ITEM(1)(1:6).EQ.'UNSYMM') THEN       
            IGTBCS= 0
c symmetrize G operator: +1 symmetrize, -1 anti-symmetrize
          ELSE IF (ITEM(1)(1:6).EQ.'SYM_G') THEN
            ISYMMET_G = +1
            DO IITEM = 2, NITEM
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.LE.2) THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &               ISYMMET_G,MXPLNC)
              END IF
            END DO
c calculate new H0, or not
          ELSE IF (ITEM(1)(1:6).EQ.'CALCH0') THEN
            I_DO_H0 = 1
            DO IITEM = 2, NITEM
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.LE.2) THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &               I_DO_H0,MXPLNC)
              END IF
            END DO
c define accuracy level
c ACCU,<expg_thrsh>,<mxterm_expg>
          ELSE IF (ITEM(1)(1:4).EQ.'ACCU') THEN
            DO IITEM = 2, NITEM
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.EQ.2) THEN
                CALL CHAR_TO_REAL(ITEM(IITEM),
     &               EXPG_THRSH,MXPLNC)
              END IF
              IF(ITEM(IITEM)(1:1).NE.' '.AND.
     &             IITEM.EQ.3) THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),
     &               MXTERM_EXPG,MXPLNC)
              END IF
            END DO
c reduce paramter space for closed-shell calculations
          ELSE IF (ITEM(1)(1:6).EQ.'CLOSED'.OR.
     &             ITEM(1)(1:6).EQ.'SPINAD') THEN
            IGTB_CLOSED=1
c dispose T-T coupled part of G:
          ELSE IF (ITEM(1)(1:4).EQ.'NOTT') THEN
            IGTB_DISPTT=1
            IGTB_CLOSED=1
c project out 0-eigenvalues of S(ij)
          ELSE IF (ITEM(1)(1:6).EQ.'PRJOUT') THEN
            IGTB_PRJOUT=1
c test H1 operator
          ELSE IF (ITEM(1)(1:6).EQ.'TESTH1') THEN
            IGTB_TEST_H1=1
c test exp(G) expansion (default)
          ELSE IF (ITEM(1)(1:5).EQ.'EXPG ') THEN
            IGTBMOD=0
c test exp(G^2) expansion
          ELSE IF (ITEM(1)(1:5).EQ.'EXPG2') THEN
            IGTBMOD=1
c test LL parameterization of G operator
          ELSE IF (ITEM(1)(1:4).EQ.'G=LL') THEN
            IGTBMOD=2
          ELSE IF (ITEM(1)(1:3).EQ.'UOU') THEN
            IGTBMOD=3
c set FUSK-level for experiments (Tak, Jeppe, for dette fantastiske ord -- ak)
          ELSE IF (ITEM(1)(1:4).EQ.'FUSK') THEN
            DO IITEM = 2, MAX(2,NITEM)
              IF(ITEM(IITEM)(1:1).NE.' ') THEN
                CALL CHAR_TO_INTEGER(ITEM(IITEM),IGTBFUSK,MXPLNC)
              END IF
            END DO            
          ELSE
            I_SUB_KW = 0
            BACKSPACE(LUIN)
          END IF  
        END DO
        GOTO 999
      END IF
*. 142: analysis (Energy along G) of GTB functional
      IF(CARD(2:7).EQ.'GTBEAG') THEN
        READ(LUIN,*) N_EAG
        ISETKW(142) = 1
        IF (N_EAG.GT.MXPANA.OR.N_EAG.LT.1) THEN
          WRITE(6,*) 'Silly input for GTBEAG! (',N_EAG,')'
          WRITE(6,*) 'First entry should be between 1 AND ',MXPANA
          NERROR = NERROR + 1
          ISETKW(142) = -1
          GOTO 999
        END IF
        DO II = 1, N_EAG
          READ(LUIN,*) IT_EAG(II)
        END DO
        READ(LUIN,*) NN_EAG
        IF (NN_EAG.GT.MXPANA.OR.NN_EAG.LT.1) THEN
          WRITE(6,*) 'Silly input for GTBEAG! (',NN_EAG,')'
          WRITE(6,*) 'The entry should be between 1 AND ',MXPANA
          NERROR = NERROR + 1
          ISETKW(142) = -1
          GOTO 999
        END IF
        DO II = 1, NN_EAG
          READ(LUIN,*) NG_EAG(II), ST_EAG(II), EN_EAG(II), NP_EAG(II)
        END DO
        GOTO 999
      END IF
*. 143: analysis (First order overlap) of GTB functional
      IF(CARD(2:7).EQ.'GTBFOO') THEN
        READ(LUIN,*) N_FOO
        ISETKW(143) = 1
        IF (N_FOO.GT.MXPANA.OR.N_FOO.LT.1) THEN
          WRITE(6,*) 'Silly input for GTBFOO! (',N_FOO,')'
          WRITE(6,*) 'First entry should be between 1 AND ',MXPANA
          NERROR = NERROR + 1
          ISETKW(143) = -1
        END IF
        DO II = 1, N_FOO
          READ(LUIN,*) IT_FOO(II)
        END DO
        GOTO 999
      END IF
*. 144: analysis (Hessian) of GTB functional
      IF(CARD(2:7).EQ.'GTBHSS') THEN
        READ(LUIN,*) N_HSS
        ISETKW(144) = 1
        IF (N_HSS.GT.MXPANA.OR.N_HSS.LT.1) THEN
          WRITE(6,*) 'Silly input for GTBHSS! (',N_HSS,')'
          WRITE(6,*) 'First entry should be between 1 AND ',MXPANA
          NERROR = NERROR + 1
          ISETKW(144) = -1
        END IF
        DO II = 1, N_HSS
          READ(LUIN,*) IT_HSS(II)
        END DO
        GOTO 999
      END IF
*. 145: Hurray, specify the amount of memory you like !!
      IF(CARD(2:7).EQ.'WRKSPC') THEN
        READ(LUIN,*) MAXMEM
        ISETKW(145) = 1
        IF (MAXMEM.LE.0) THEN
          WRITE(6,*) 'Silly input for WRKSPC! (',MAXMEM,')'
          NERROR = NERROR + 1
          ISETKW(145) = -1
        END IF
        GOTO 999
      END IF
*. 146: Experimental switch to avoid the allocation of N^4 arrays
*.       in the very beginning
      IF(CARD(2:7).EQ.'SAVMEM') THEN
        ISETKW(146) = 1
        ISVMEM=1
        GOTO 999
      END IF
*. 147: Define the target wave-function model
*        needed for passing information to subsequent programs
*        i.pt. for geom. gradient calculations
*        read one more line containing the space and the number of the
*        calculation within that space
      IF(CARD(2:7).EQ.'TARGET') THEN
        ISETKW(147) = 1
        READ(LUIN,*) ITGSPC,ITGCLC
        GOTO 999
      END IF
*
*. 148: Generate cumulant matrices throgh construction of 
*.       normal density matrices 
      IF(CARD(2:7).EQ.'CUMULA') THEN
*. Read max order of cumulants to be constructed 
        READ(5,*) ICUMULA
        ISETKW(148) = 1
        GOTO 999
      END IF
*
*. 149: Restart IC calculation 
*
      IF(CARD(2:7).EQ.'RSTRIC') THEN
*. Restart internal contraction calculation 
        IRESTRT_IC = 1
        ISETKW(149) = 1
        GOTO 999
      END IF
*
*. 150: Number of commutators in MRCC calculations
*        Specify numbers for three types of commutators: 
*        NCOMMU_E: Number of commutators in energy evaluation and
*                   reoptimization of internal state
*        NCOMMU_J: Number of commutators used in 
*                   approximate Jacobian 
*        NCOMMU_V: Number of commutators in CC vector function
      IF(CARD(2:7).EQ.'NCOMMU') THEN
        READ(5,*) NCOMMU_E, NCOMMU_J, NCOMMU_V
        ISETKW(150) = 1
        GOTO 999
      END IF
*
*. 151: Approximate higher commutator in MRCC energy function 
*
      IF(CARD(2:7).EQ.'APRCME') THEN
        I_APPROX_HCOM_E = 1
        ISETKW(151) = 1
        GOTO 999
      END IF
*
*. 152: Approximate higher commutator in MRCC vector function 
*
      IF(CARD(2:7).EQ.'APRCMV') THEN
        I_APPROX_HCOM_V = 1
        ISETKW(152) = 1
        GOTO 999
      END IF
*
*. 153: Approximate higher commutator in Approximate Jacobian
*
      IF(CARD(2:7).EQ.'APRCMJ') THEN
        I_APPROX_HCOM_J = 1
        ISETKW(153) = 1
        GOTO 999
      END IF
*
* 154: In which orbital spaces should density be calculated in ?
      IF(CARD(2:7).EQ.'DENSPC') THEN
*. Three numbers, IDENS_IN,IDEN_AC,IDENS_SEC
*. IDENS_IN = 1 => Construct density in inactive space
*. IDENS_AC = 1 => Construct density in active space 
*. IDENS_SEC = 1 => Construct density in secondary space 
        READ(LUIN,*) IDENS_IN, IDENS_AC, IDENS_SEC
        ISETKW(154) = 1
        GOTO 999
      END IF
*
*.155:  Read in initial S and J from DISC file LU_SJ
*
      IF(CARD(2:7).EQ.'READSJ') THEN
        IREADSJ = 1
        ISETKW(155) = 1
        GOTO 999
      END IF
*
*. 156: Use product expansion  of wave-functions
*
      IF(CARD(2:7).EQ.'PRDEXP') THEN
        I_DO_PRODEXP = 1
        ISETKW(156) = 1
        GOTO 999
      END IF
*
*. 157: The wave-functions for each orbital subspace/atom for product wave functions
*
      IF(CARD(2:7).EQ.'PRDWVF') THEN
*. Input goes as 
*. Loop over orbital subspaces
*.   Read in number of subspace wave functions for this orbital subspace
*    Loop over the subspace wave functions for this orbital subspace 
*    Read in occupation and total spin of each subspace wave function
*.   ( It could be that spin-patterns should also be added)
*. GASSH should be given before to give the info on the total number of GASpaces 
*. and the number of orbital in each subspace
*. Check first whether GASSH has been defined 
         IF(ISETKW(50).EQ.0) THEN
           WRITE(6,*) ' Dear User'
           WRITE(6,*)
           WRITE(6,*) ' GASSH must be specified before PRDWVF'
           WRITE(6,*)
     &     ' Else I do not know about the number of orbital spaces'
           WRITE(6,*) ' So I will stop '
           STOP 'READIN: put GASSH before PRDWVF'
         END IF
         DO IGAS = 1, NGAS
           READ(5,*) NWF_PER_SUBSPC(IGAS)
           DO IWF = 1, NWF_PER_SUBSPC(IGAS)
             READ(5,*)(ISUBSPCWF_OCC(IORB,IWF,IGAS),IORB=1,NOBPT(IGAS)),
     &                MULT_FOR_SUBSPCWF(IWF,IGAS)
           END DO
         END DO
         ISETKW(157) = 1
         GOTO 999
      END IF
*
*. 158: Max ineratomic excitation level for product wf expansion
*
      IF(CARD(2:7).EQ.'PRDEXC') THEN
*. Read in allowed excitation level between different atoms in product wave function 
*. expansion  
        READ(5,*) INTRA_EXC_PRWF
        ISETKW(158) = 1
        GOTO 999
      END IF
*
* 159: Printflag for CSF information
      IF(CARD(2:7).EQ.'IPRCSF') THEN
        READ(5,*) IPRCSF
        ISETKW(159) = 1
        GOTO 999
      END IF
*
* 160: Include active-active excitations 
*
      IF(CARD(2:7).EQ.'INC_AA') THEN
        I_INC_AA = 1
        ISETKW(160) = 1
        WRITE(6,*) ' Input: I_INC_AA = ', I_INC_AA
        GOTO 999
      ENDIF
*
* 161: Threshold for singularities of metric in IN
*
      IF(CARD(2:6).EQ.'SINGU') THEN
        READ(5,*) THRES_SINGU
        ISETKW(161) = 1
C?      WRITE(6,*) ' THRES_SINGU = ', THRES_SINGU
        GOTO 999
      END IF
*
* 162: Largest number in initial iterations - for restartinf
*
      IF(CARD(2:7).EQ.'MXVC_I') THEN
        READ(5,*) MXVC_I
        ISETKW(162) = 1
C       WRITE(6,*) ' READIN: MXVC_I = ', MXVC_I
        GOTO 999
      END IF
*
* 163: Largest number of macro-iterations
*
      IF(CARD(2:7).EQ.'MXIT_M') THEN
        READ(5,*) MAXITM
        ISETKW(163) = 1
        GOTO 999
      END IF
*
* 164: Freeze internal states in internal contraction calc?
*
      IF(CARD(2:7).EQ.'FR_INT') THEN
        I_FIX_INTERNAL = 1
        ISETKW(164) = 1
        GOTO 999
      END IF
*
* 165: Form of Hamiltonian used to define internal zero order states
*
      IF(CARD(2:7).EQ.'ZS_HAM') THEN
        READ(5,*) I_INT_HAM
        ISETKW(165) = 1
        GOTO 999
      END IF
*
* 166: Form of Hamiltonian used to define internal zero order states
*
*. Use internal contraction without EI-split
      IF(CARD(2:6).EQ.'NO_EI') THEN
        I_DO_EI = 0
        IEI_VERSION = 0
        ISETKW(166) = 1
        GOTO 999
      END IF
*
* 167: Read in excitation information in form appropriate for 
*      internal contaction CI from f.ex. SD reference
*
       IF(CARD(2:7).EQ.'IC_EXO') THEN
         READ(5,*) ICEXC_RANK_MIN,ICEXC_RANK_MAX, ICEXC_INT
         ISETKW(167) = 1
         GOTO 999
       END IF
*
* 168: General internal contraction operators for GIC
*
       IF(CARD(2:7).EQ.'GIC_EX') THEN
*. Number of external T-operators
         READ(5,*) NTEXC_G
         DO ITEXC = 1, NTEXC_G
*. Min exc rank, max exc. ran, are internal allowed, space to be
*. projected out, space of resulting operator
           READ(5,*) ICEXC_RANK_MIN_G(ITEXC),ICEXC_RANK_MAX_G(ITEXC),
     &            ICEXC_INT_G(ITEXC), IPTCSPC_G(ITEXC),ITCSPC_G(ITEXC)
         END DO
         ISETKW(168) = 1
         GOTO 999  
       END IF
*
* 169: Print level for MCSCF
*
       IF(CARD(2:7).EQ.'IPRMCS') THEN
         READ(5,*)  IPRMCSCF
         ISETKW(169) = 1
         GOTO 999
       END IF
*
* 170: Method for optimizing MCSCF wave function
*
       IF(CARD(2:7).EQ.'MCSCFA') THEN
*. New (october 2011: IMCSCF_MET, IOOE2_APR, I_DO_LINSEA_MCSCF
       READ(LUIN,'(A)') CARD1
       CALL LFTPOS(CARD1,MXPLNC)
       CALL UPPCAS(CARD1,MXPLNC)
       CALL DECODE_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
*. The three expected input parameters:
* IMCSCF_MET = 2 => Two-step Newton procedure 
*            = 1 => One-step Newton procedure (not implemented pt)
*            = 3 => Two-step Update
*            = 4 => One-step update
*
* IOOE2_APR  = 1 => Construct full orbital Hessian
*            = 2 => Construct diagonal blocks of orbital Hessian (not implemented)
*            = 3 => Construct approximate diagonal (not implemented)
* I_DO_LINSEA_MCSCF = 1 => Perform Line search in orbital optimization
*                   = 0 => Perform line search if energy increases in
*                          orbital optimization
*                   = -1 => Never perform line search
         IF(NITEM.NE.3) THEN
           WRITE(6,*) 
     &     ' 3 entries required in MCSCFA: ',
     &     ' IMCSCF_MET, IOOE2_APR,I_DO_LINSEA_MCSCF'
           WRITE(6,*) ' Number of items supplied by you ', NITEM
           ISETKW(170) = -1
           NERROR = NERROR + 1
         ELSE
*. Three items supplied, proceed as 
           CALL CHAR_TO_INTEGER(ITEM(1),IMCSCF_MET,MXPLNC)
           CALL CHAR_TO_INTEGER(ITEM(2),IOOE2_APR,MXPLNC)
           CALL CHAR_TO_INTEGER(ITEM(3),I_DO_LINSEA_MCSCF,MXPLNC)
           ISETKW(170) = 1
           IF(IMCSCF_MET.LT.1.OR.IMCSCF_MET.GT.4) THEN
             WRITE(6,*) ' Illegal value of IMCSCF_MET = ', IMCSCF_MET
             ISETKW(170) = -1
           END IF
           IF(IOOE2_APR.LT.1.OR.IOOE2_APR.GT.3) THEN
             WRITE(6,*) ' Illegal value of IOOE2_APR = ', IOOE2_APR
             ISETKW(170) = -1
           END IF
           IF(I_DO_LINSEA_MCSCF.LT.-1.OR.I_DO_LINSEA_MCSCF.GT.1) THEN
             WRITE(6,*) ' Illegal value of I_DO_LINSEA_MCSCF = ', 
     &       I_DO_LINSEA_MCSF
             ISETKW(170) = -1
           END IF
         END IF
         GOTO 999
       END IF
*
* 171: Form of storing and transforming two-electron integrals 
*
       IF(CARD(2:7).EQ.'TRA_RO') THEN
         READ(5,*) ITRA_ROUTE
* ITRA_ROUTE = 1 => Old storage and transformation
* ITRA_ROUTE = 2 => New storage and transformation
         IF(ITRA_ROUTE.NE.1.AND.ITRA_ROUTE.NE.2) THEN
           WRITE(6,*) ' Illegal value of ITRA_ROUTE = ', ITRA_ROUTE
           WRITE(6,*) ' Allowed values: 1, 2 '
           ISETKW(171) = -1
           NERROR = NERROR + 1
         END IF
         ISETKW(171) = 1
         GOTO 999
       END IF
*
* 172: Info in nonorthogonal calculation. Pt calculation is 
*      required to have NGAS-1 othogonal spaces and one 
*      nonorthogonal space, NORTCI_SCVB_SPACE.  The CI expansion in the 
*      nonorthogonal is NORTCI_SCVP_EXCIT excitation out from a 
*      standard spin-coupled valence bond calculation
       IF(CARD(2:7).EQ.'NORTIN') THEN
*. Read two numbers: Nonorthogonal space and allowed number of 
*. excitations out from spin coupled valence bond
         READ(5,*)  NORTCIX_SCVB_SPACE,NORTCI_SCVB_EXCIT
C?       WRITE(6,*) ' Test: NORTCIX_SCVB_SPACE,NORTCI_SCVB_EXCIT = ',
C?   &                      NORTCIX_SCVB_SPACE,NORTCI_SCVB_EXCIT
         ISETKW(172) = 1 
         GOTO 999
       END IF
*
       IF(CARD(2:7).EQ.'VBRFSP') THEN
*. Info in valence bond CI reference space
* Max and min occupation in CI reference space
*  
*. requires: NORTCIX: To define Nonoorthogonal orbital space
*            GASSH - to give info on number of orbitals in space
         IERROR_LOC = 0
         IF(ISETKW(172).NE.1) THEN
           WRITE(6,*) ' NORTCIX keyword must be specified before VBRFSP'
           NERROR = NERROR + 1
           IERROR_LOC = 1
         END IF
         IF(ISETKW(50).NE.1) THEN
           WRITE(6,*) ' GASSH keyword must be specified before  VBRFSP'
           IERROR_LOC = 1
           NERROR = NERROR + 1
         END IF
         IF(IERROR_LOC.EQ.1) THEN
           ISETKW(173) = -1 
           GOTO 999
         ELSE
           NORBVBSPC = NOBPT(NORTCIX_SCVB_SPACE)
C?         WRITE(6,*) ' NOBPT(1), NOBPT(2) = ', NOBPT(1), NOBPT(2)
           WRITE(6,*) ' Test: NORTCIX_SCVB_SPACE, NORBVBSPC = ',
     &                        NORTCIX_SCVB_SPACE, NORBVBSPC 
*. Max and min occupation in nonorthogonal CI space
           READ(5,*) (VB_REFSPC_MIN(IORB),IORB = 1, NORBVBSPC)
           READ(5,*) (VB_REFSPC_MAX(IORB),IORB = 1, NORBVBSPC)
           CALL ICOPVE(VB_REFSPC_MIN,VB_REFSPCO_MIN,NORBVBSPC)
           CALL ICOPVE(VB_REFSPC_MAX,VB_REFSPCO_MAX,NORBVBSPC)
           ISETKW(173) = 1
           GOTO 999
         END IF
       END IF
*
* 174: Method for performing Northogonal CI calculation
*
       IF(CARD(2:7).EQ.'NORT_M') THEN
* NORT_MET = 1 => Reexpand in full space
* NORT_MET = 2 => Configuration based approach
         READ(5,*)  NORT_MET
         ISETKW(174) = 1
         GOTO 999
       END IF
*
* 175: Information on MO fragments of molecule
*
       IF(CARD(2:7).EQ.'MOFRAG') THEN
*. Number of different types of fragments: say number of different 
*. types of atoms
         READ(5,*) NFRAG_TP
*. Characters for the Fragments (Character*3 )
          READ(5,*) (CFRAG(IFRAG),IFRAG = 1, NFRAG_TP)
*. Number of fragments making up the molecule, two for a diatomic 
*  if each atom  is a fragment
         READ(5,*) NFRAG_MOL
*. And read in the fragments making up the molecule. 
*. Should be in the order of the MOLECULE.INP file
         READ(5,*) (IFRAG_MOL(IFRAG),IFRAG=1, NFRAG_MOL)
         ISETKW(175) = 1
         GOTO 999
       END IF
*
* 176: Information about mapping of fragment MO's to molecule MO's
*
       IF(CARD(2:7).EQ.'FRAGOB') THEN
*. Call routine to readin fragment to orbital info
         CALL ASSEMBLE_MO_FROM_FRAGMENTS_DEFINE
         ISETKW(176) = 1
         GOTO 999
       END IF
*
* 177: Information on generation of initial orbitals for nonorthogonal
*      calculation
*
        IF(CARD(2:7).EQ.'INI_MO') THEN
          READ(5,*)  INI_MO_TP, INI_MO_ORT, INI_ORT_VBGAS
          IF(IECHO.EQ.1) THEN
            WRITE(6,'(A,3I4)') 
     &      ' READIN: INI_MO_TP, INI_MO_ORT,  INI_ORT_VBGAS  = ',
     &                INI_MO_TP, INI_MO_ORT,  INI_ORT_VBGAS
          END IF
          ISETKW(177) = 1
*. INI_MO_TP: Construction of the various GAS spaces
*            = 1 => Use atomic orbitals as orbitals
*            = 2 => Rotate orbitals in MOAOIN (from DALTON) calc
*                   so the MOAO matrix is diagonal in VB active space
*            = 3 => Use MOAOIN orbitals without modifications
*            = 4 => Built info from fragment information
*            = 5 => read in from LUCINF_O and orthogonalize 
*                    as specified by INI_MO_ORT
* INI_MO_ORT: Othogonalization in Orbital subspaces
*            = 0 => No orthogonalization
*            = 1 => Symmetric orthogonalization
*            = 2 => Diagonalize metric
* INI_ORT_VBGAS = 1: Orthogonalize GASpace of VB calculation
*            = 0: No orthogonalization of GASpace for VB calc.
          IF(INI_MO_TP.LT.1.OR.INI_MO_TP.GT.5) THEN
            WRITE(6,*) ' INI_MO_TP out of range, value = ', INI_MO_TP
            ISETKW(177) = -1
            NERROR = NERROR + 1
          END IF
          IF(INI_MO_ORT.LT.0.OR.INI_MO_ORT.GT.2) THEN
            WRITE(6,*) ' INI_MO_ORT out of range, value = ',INI_MO_ORT
            ISETKW(177) = -1
            NERROR = NERROR + 1
          END IF
          IF(INI_ORT_VBGAS.LT.0.OR.INI_ORT_VBGAS.GT.1) THEN
            WRITE(6,*) 
     &      ' INI_ORT_VBGAS out of range, value = ',INI_ORT_VBGAS
            ISETKW(177) = -1
            NERROR = NERROR + 1
          END IF
          GOTO 999
        END IF
*
* 178: Read in initial configuration: Ascending list of occupied orbitals
*                                in active space
*
* 2 electrons in orbital 3, 1 electron in orbital 5,
* 1 electron in orbital 6 is therefore specified as  3 3 5 6
        IF(CARD(2:7).EQ.'INICNF') THEN
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
           CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
*. And as integers
           NELEC_INICONF = NITEM
           DO IELEC = 1, NELEC_INICONF
                CALL CHAR_TO_INTEGER(ITEM(IELEC),
     &               INT_ITEM(IELEC),MXPLNC)
           END DO
*. Reform to standard compressed form
C REFORM_CONF_OCC(IOCC_EXP,IOCC_PCK,NEL,NOCOB,IWAY)
           NOCOB_L = 0
           CALL REFORM_CONF_OCC(INT_ITEM,INI_CONF,
     &          NELEC_INICONF,NOCOB_L,1)
           NOB_INI_CONF = NOCOB_L
           IF(IECHO.EQ.1) THEN
             WRITE(6,*) '  Initial Configuration: '
             CALL IWRTMA(INI_CONF,1,NOB_INI_CONF,1,NOB_INI_CONF)
           END IF
           ISETKW(178) = 1
           I_HAVE_INI_CONF = 1
           GOTO 999
        END IF
*
* 179: Print flag for Valence bond calculation
*
        IF(CARD(2:5).EQ.'PRVB') THEN
          READ(5,*) IPRVB
          ISETKW(179) = 1
          GOTO 999
        END IF
*
* 180: GIO method: Specify occupation of space
*      where Hamiltonian is calculated exact
*
        IF(CARD(2:7).EQ.'GIOMET') THEN
*. Read in min and max for space where Hamiltonian 
* is calculated exact
         READ(5,*) (IOCCPSPC(IGAS,1),IGAS=1,NGAS)
         READ(5,*) (IOCCPSPC(IGAS,2),IGAS=1,NGAS)
         ISETKW(180) = 1 
         I_AM_GIOVANNI = 1
         GOTO 999
        END IF
*
* 181: Print flag for integrals
*
        IF(CARD(2:7).EQ.'IPRINT') THEN
          READ(5,*) IPRINTEGRAL
          ISETKW(181) = 1
          GOTO 999
        END IF
*
* 182: Form of preconditioner in CSF basis
*
       IF(CARD(2:7).EQ.'H0_CSF') THEN
* 1 => Averaged determinant diagonal
* 2 => CSF diagonal
* 3 => Configuration blocks of H
        READ(5,*)  IH0_CSF
        ISETKW(182) = 1
        GOTO 999
       END IF
*
* 183: Form of batchning of configurations
*
      IF(CARD(2:7).EQ.'CNFBAT') THEN
       READ(5,*) ICNFBAT
* ICNFBAT = 1 => Complete configuration expansion  stored in memory
* ICNFBAT = 2 => Configuration info and expansions stored for a 
*                single OCCLS
        ISETKW(183) = 1
        GOTO 999
      END IF
*
*. 184: Combine several GASpaces into ensembles
*
      IF(CARD(2:7).EQ.'ENSMGS') THEN
        READ(5,*) NENSGS
        DO JENSGS = 1, NENSGS
*. Number of GASpaces in this ensemble
          READ(5,*) LENSGS(JENSGS)
*. And the GASpaces of this ensemble
          READ(5,*) (IENSGS(IGAS,JENSGS),IGAS = 1, LENSGS(JENSGS))
        END DO
        ISETKW(184) = 1
        GOTO 999
      END IF
*
*. 185: Constraints on occupation in a ensemble space
*
      IF(CARD(2:7).EQ.'ENSCON') THEN
*. Constraints are imposed only on Ensemble GAS 1
*. Requires that the number number of CI spaces have been defined
        IF(ISETKW(51).EQ.0) THEN
          WRITE(6,*) 
     &    ' Keyword ENSCON requires the number of CISPACES'
          WRITE(6,*) 
     &    ' Specify GASSPC before ENSCON '
          ISETKW(185) = -1
          NERROR = NERROR + 1
          GOTO 999
        ELSE
*. Number of allowed values of electrons in Ensemble gas space 
          I_CHECK_ENSGS = 1
          DO ISPC = 1, NCISPC
            READ(5,*) NELVAL_IN_ENSGS(ISPC)
            READ(5,*) (IEL_IN_ENSGS(IEL,ISPC),
     &                 IEL = 1, NELVAL_IN_ENSGS(ISPC))
          END DO
          ISETKW(185) = 1
          GOTO 999
        END IF
      END IF
*
* 186: Symmetry-equivalent groups of fragments
*
       IF(CARD(2:7).EQ.'EQFRAG') THEN
          ISETKW(186) = 1
*. Read in symmetry-equivalent sets of fragments 
          READ(5,*) NEQVGRP_FRAG
          DO IEQV = 1, NEQVGRP_FRAG
            READ(5,*) LEQVGRP_FRAG(IEQV)
            READ(5,*) 
     &      (IEQVGRP_FRAG(IFRAG,IEQV),IFRAG = 1, LEQVGRP_FRAG(IEQV))
          END DO
          GOTO 999
      END IF
*
* 187: Supersymmetry in use
*
      IF(CARD(2:7).EQ.'SUPSYM') THEN
        ISETKW(187) = 1
        I_USE_SUPSYM = 1
*. Read form of supersymmetry
        READ(LUIN,'(A)') CARD1
        CALL LFTPOS(CARD1,MXPLNC)
        CALL UPPCAS(CARD1,MXPLNC)
*
        IF(CARD1(1:6).EQ.'ATOMIC') THEN
*. Atomic supersymmetry
           CSUPSYM(1:6) = 'ATOMIC'
*. No explicit use of inversion symmetry so
           INVCNT = 0
        ELSE IF (CARD1(1:6).EQ.'LINEAR') THEN
           CSUPSYM(1:6) = 'LINEAR'
*. For linear there are two possibilities: DinfH or CinfV.
*. Look on number of irreps to devide whether there is a center of inversion
           IF(NIRREP.EQ.8) THEN
            INVCNT = 1
           ELSE 
            INVCNT = 0
           END IF
        ELSE
          WRITE(6,*) ' Illegal form of SUPSYM :'  
          INI_HF_MO = 0
          WRITE(6,'(A,A)') ' Your suggestion: ', CARD1     
          WRITE(6,*) ' Allowed entries: '
          WRITE(6,*) ' =================='
          WRITE(6,*)    'ATOMIC' 
          WRITE(6,*)    'LINEAR'
          NERROR = NERROR + 1
          ISETKW(187) = -1
        END IF
        GOTO 999
      END IF
*
* 188 Double occupied super-symmetry irreps for HF
*     is also used for HF occupations for standard
*
      IF(CARD(2:7).EQ.'HFD_OC') THEN
C?      WRITE(6,*) ' HFD_OC identified '
        ISETKW(188) = 1
*. Read line in, number of irreps not known 
        IZERO = 0
        CALL ISETVC(NHFD_IRREP_SUPSYM,IZERO, MAX_SUPSYM_IRREP)
        READ(LUIN,'(A)') CARD1
        CALL LFTPOS(CARD1,MXPLNC)
        CALL UPPCAS(CARD1,MXPLNC)
        CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
        IF(NITEM.GT.MAX_SUPSYM_IRREP) THEN
          WRITE(6,*) ' HFD_SP: Number of irreps larger than allowd'
          WRITE(6,*) ' Actual and MAX: ', NITEM, MAX_SUPSYM_IRREP
          NERROR = NERROR + 1
          ISETKW(188) = -1
*. Reduce temporary number of irreps
          NITEM = MAX_SUPSYM_IRREP
        END IF
        NACT_SUPSYM_IRREP = MAX(NACT_SUPSYM_IRREP,NITEM)
*. And change into integers and save 
        DO IRREP = 1, NITEM
          CALL CHAR_TO_INTEGER(ITEM(IRREP),
     &         NHFD_IRREP_SUPSYM(IRREP),MXPLNC)
        END DO
        GOTO 999
      END IF
*
* 189 Singly occupied super-symmetry irreps for HF
*
      IF(CARD(2:7).EQ.'HFS_OC') THEN
        ISETKW(189) = 1
*. Read line in, number of irreps not known 
        IZERO = 0
        CALL ISETVC(NHFS_IRREP_SUPSYM,IZERO, MAX_SUPSYM_IRREP)
        READ(LUIN,'(A)') CARD1
        CALL LFTPOS(CARD1,MXPLNC)
        CALL UPPCAS(CARD1,MXPLNC)
        CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
        IF(NITEM.GT.MAX_SUPSYM_IRREP) THEN
          WRITE(6,*) ' HFS_SP: Number of irreps larger than allowd'
          WRITE(6,*) ' Actual and MAX: ', NITEM, MAX_SUPSYM_IRREP
          NERROR = NERROR + 1
          ISETKW(189) = -1
*. Reduce temporary number of irreps
          NITEM = MAX_SUPSYM_IRREP
        END IF
        NACT_SUPSYM_IRREP = MAX(NACT_SUPSYM_IRREP,NITEM)
*. And change into integers and save 
        DO IRREP = 1, NITEM
          CALL CHAR_TO_INTEGER(ITEM(IRREP),
     &         NHFS_IRREP_SUPSYM(IRREP),MXPLNC)
        END DO
        GOTO 999
      END IF
*
* 190 super-symmetry irreps for GAS including inactive and secondary
*
      IF(CARD(2:7).EQ.'GAS_SP') THEN
        ISETKW(190) = 1
        IZERO = 0
        DO IGAS = 0, NGAS + 1
*. Read line in, number of irreps not known 
          CALL ISETVC(NGAS_IRREP_SUPSYM(1,IGAS),IZERO, MAX_SUPSYM_IRREP)
          READ(LUIN,'(A)') CARD1
          CALL LFTPOS(CARD1,MXPLNC)
          CALL UPPCAS(CARD1,MXPLNC)
          CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
          IF(NITEM.GT.MAX_SUPSYM_IRREP) THEN
            WRITE(6,*) ' GAS_SP: Number of irreps larger than allowd'
            WRITE(6,*) ' Actual and MAX: ', NITEM, MAX_SUPSYM_IRREP
            NERROR = NERROR + 1
            ISETKW(188) = -1
*. Reduce temporary number of irreps
            NITEM = MAX_SUPSYM_IRREP
          END IF
          NACT_SUPSYM_IRREP = MAX(NACT_SUPSYM_IRREP,NITEM)
*. And change into integers and save 
          DO IRREP = 1, NITEM
            CALL CHAR_TO_INTEGER(ITEM(IRREP),
     &           NGAS_IRREP_SUPSYM(IRREP,IGAS),MXPLNC)
          END DO
        END DO
        GOTO 999
      END IF
*
      IF(CARD(2:7).EQ.'NOSPFI') THEN
*. Final orbitals are not ordered using super-symmetry
       ISETKW(191) = 1
       I_NEGLECT_SUPSYM_FINAL_MO = 1
       GOTO 999
      END IF
*
      IF(CARD(2:7).EQ.'FRG=LU') THEN
*. Fragments are read from LUCIA files - even if normal environment is DALTON
       ISETKW(192) = 1
       I_USE_LUCIA_FRAGMENTS = 1
       GOTO 999
      END IF
*
      IF(CARD(2:7).EQ.'FRZORB') THEN
*. Orbitals frozen in MCSCF optimization  -are given in Type order
        READ(LUIN,'(A)') CARD1
        CALL LFTPOS(CARD1,MXPLNC)
        CALL UPPCAS(CARD1,MXPLNC)
        CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
        NFRZ_ORB = NITEM
        DO IORB = 1, NFRZ_ORB
          CALL CHAR_TO_INTEGER(ITEM(IORB),IFRZ_ORB(IORB),MXPLNC)
        END DO
        ISETKW(193) = 1
        GOTO 999
      END IF
      IF(CARD(2:7).EQ.'SBSPPR') THEN
*. Subspace in which exact Hamiltonian will be used for preconditioner and 
*. first space where this will be applied
        READ(5,*)  ISBSPPR, ISBSPPR_INI
        ISETKW(194) = 1
        GOTO 999
      END IF
*
*
*. Number of roots in initial CI
*
      IF(CARD(2:7).EQ.'IN_NRO') THEN
       READ(5,*)  INI_NROOT
       ISETKW(195) = 1
       GOTO 999
      END IF
*
*. Root to be selected from initial CI
*
COLD  IF(CARD(2:7).EQ.'IN_SRO') THEN
*. Specify a method + value:
*  SELORD, INI_SROOT => Select root number INI_SROOT
*  SELSPS , INI_SUPSYM => Select root with super-symmetry INI_SUPSYM
COLD   READ(5,*)  INI_ROOTM, IVAL
COLD   WRITE(6,*) ' INI_ROOTM, IVAL = ', INI_ROOTM, IVAL
COLD   ISETKW(196) = 1
COLD   INI_SROOT = 0
COLD   IF(INI_ROOTM(1:6).EQ.'SELORD') THEN
COLD     INI_SROOT = IVAL
COLD   ELSE IF (INI_ROOTM(1:6).EQ.'SELSPS') THEN
COLD     INI_SUPSYM = IVAL
COLD   ELSE
COLD     WRITE(6,*) ' Illegal string in IN_SRO ', INI_ROOTM
COLD     NERROR = NERROR + 1
COLD     ISETKW(196) = -1
COLD   END  IF
COLD   WRITE(6,*) ' From input, INI_SUPSYM = ', INI_SUPSYM
COLD   GOTO 999
COLD  END IF
C
      IF(CARD(2:7).EQ.'SEL_RT') THEN
*. Specify a method + value:
*  SELORD, INI_SROOT => Select root number INI_SROOT
*  SELSPS , INI_SUPSYM => Select root with super-symmetry INI_SUPSYM
       READ(5,*)  IROOT_MET, IVAL
       WRITE(6,*) ' IROOT_MET, IVAL = ', IROOT_MET, IVAL
       ISETKW(196) = 1
       ITG_SROOT = 0
       IF(IROOT_MET(1:6).EQ.'SELORD') THEN
         ITG_SROOT = IVAL
       ELSE IF (IROOT_MET(1:6).EQ.'SELSPS') THEN
         ITG_SUPSYM = IVAL
       ELSE
         WRITE(6,*) ' Illegal string in SEL_RT ', IROOT_MET
         NERROR = NERROR + 1
         ISETKW(196) = -1
       END  IF
       WRITE(6,*) ' From input, ITG_SUPSYM = ', ITG_SUPSYM
*. Should this root selection be carried out in all calculations or just initial?
       READ(5,*) CARD1
       CALL LFTPOS(CARD1,MXPLNC)
       CALL UPPCAS(CARD1,MXPLNC)
       IF(CARD1(1:3).EQ.'INI') THEN
         ISEL_ONLY_INI = 1
       ELSE IF(CARD1(1:3).EQ.'ALL') THEN
         ISEL_ONLY_INI = 0
       ELSE 
         WRITE(6,*) ' Unknown value for ISEL_ONLY_INI'
         WRITE(6,*) ' Allowed input: INI ALL '
         WRITE(6,*) ' Given input ', CARD1
         NERROR = NERROR + 1
         ISETKW(196) = -1
       END IF
       GOTO 999
      END IF
*
* Root selection (during optimization sequence)
*
      IF(CARD(2:7).EQ.'RT_SEL') THEN
        READ(5,*) IROOT_SEL
        ISETKW(197) = 1
        GOTO 999
      END IF
*
* First space where specified orbitals will be frozen
*
      IF(CARD(2:7).EQ.'FRZFST') THEN
        READ(5,*)  IFRZFST
        ISETKW(198) = 1
        GOTO 999
      END IF
*
* A set of general VB spaces
*
       IF(CARD(2:7).EQ.'VBGNSP') THEN
*
* Info on additional set of MIN max spaces for nonorthogonal CI
*  
*. requires: NORTCIX: To define Nonoorthogonal orbital space
*            GASSH - to give info on number of orbitals in space
         IERROR_LOC = 0
         IF(ISETKW(172).NE.1) THEN
           WRITE(6,*) ' NORTCIX keyword must be specified before VBGNSP'
           NERROR = NERROR + 1
           IERROR_LOC = 1
         END IF
         IF(ISETKW(50).NE.1) THEN
           WRITE(6,*) ' GASSH keyword must be specified before  VBGNSP'
           IERROR_LOC = 1
           NERROR = NERROR + 1
         END IF
         IF(IERROR_LOC.EQ.1) THEN
           ISETKW(173) = -1 
           GOTO 999
         ELSE
           NORBVBSPC = NOBPT(NORTCIX_SCVB_SPACE)
*. Number of additional spaces
           READ(5,*)  NVBGNSP
           DO IVBGNSP = 1, NVBGNSP
*. Max and min occupation in nonorthogonal CI space
             READ(5,*) (VB_GNSPC_MIN(IORB,IVBGNSP),IORB = 1, NORBVBSPC)
             READ(5,*) (VB_GNSPC_MAX(IORB,IVBGNSP),IORB = 1, NORBVBSPC)
             ISETKW(199) = 1
           END DO
           GOTO 999
         END IF
       END IF
*
       IF(CARD(2:7).EQ.'VBOBOR') THEN
*. Read in order or active orbitals corresponding to the MINMAX 
* occupations in VBRFSP, VBGNSP. It is assumed that NORBVBSPC has
* been determined 
*. To be precise: The original orbital number of each reordered orbital
*. is read on
        READ(5,*)  (IREO_MNMX_OB_NO(IORB), IORB = 1,NORBVBSPC)
        WRITE(6,*) ' IREO_MNMX_OB_NO: '
        CALL IWRTMA(IREO_MNMX_OB_NO,1,NORBVBSPC ,1,NORBVBSPC)
*. And define the reverse ordering from original numbers to new
        DO IOB = 1, NORBVBSPC 
          IREO_MNMX_OB_ON(IREO_MNMX_OB_NO(IOB)) = IOB
        END DO
        WRITE(6,*) ' IREO_MNMX_OB_ON: '
        CALL IWRTMA(IREO_MNMX_OB_ON,1,NORBVBSPC ,1,NORBVBSPC)
        ISETKW(200) = 1
        GOTO 999
       END IF
*
       IF(CARD(2:7).EQ.'VBSCOR') THEN
*. Read in the order in which the open orbitals should be coupled in the 
*. Nort calculations
        READ(5,*)  (IREO_SPCP_OB_NO(IORB), IORB = 1,NORBVBSPC)
*. And determine the reverse ordering from original numbers to new
        DO IOB = 1, NORBVBSPC 
          IREO_SPCP_OB_ON(IREO_SPCP_OB_NO(IOB)) = IOB
        END DO
        ISETKW(201) = 1
        GOTO 999
       END IF
*
       IF(CARD(2:7).EQ.'AKBKME') THEN
*. Readin method for performing AKBK calculations
*  IAKBK_MET = 1 => Q-vectors in DISC
*            = 2 => No Q-vectors in DISC
         READ(5,*) IAKBK_MET
         ISETKW(202) = IAKBK_MET
         IF(IAKBK_MET.NE.1.AND.IAKBK_MET.NE.2) THEN
           WRITE(6,*) ' Illegal value of IAKBK_MET = ', IAKBK_MET
           WRITE(6,*) ' Allowed values = 1, 2 '
           ISETKW(202) = -1
           NERROR = NERROR + 1 
         END IF
         GOTO 999
       END IF
*. KEYWORD was not identified
*
       WRITE(6,*)
     & '  ****  Error, unidentified KEYWORD in READIN   **** '
       WRITE(6,*)
       WRITE(6,*) ' Last line read  '
       WRITE(6,*) ' ================'
       WRITE(LUOUT,'(10X,A)') CARD      
       WRITE(6,*)
       WRITE(6,*) ' Preceeding KEYWORD'
       WRITE(6,*) ' ==================='
       WRITE(LUOUT,'(10X,A)') LASTCARD      
       NERROR = NERROR + 1
*
      END IF! Card was not eof or comment
  999  CONTINUE
       IF(IECHO.EQ.1)
     & WRITE(6,'(A,A)') ' processed KEYWORD/COMMENT : ', CARD
*. Save previous keyword
       LASTCARD(1:72) = CARD(1:72)
      GOTO 1000
*.End of loop over KEYWORDS
 1001 CONTINUE
*
 
      IF(NERROR.NE.0) THEN
        WRITE(LUOUT,'(A)')
     &  ' Run will be aborted due to input errors '
        WRITE(LUOUT,'(A,I9)')
     &  ' Number of input errors detected in READIN ', NERROR
*
        WRITE(6,*) ' The following keywords were correctly identified'
        WRITE(6,*) ' ================================================'
        DO  IENTRY = 1, MXPKW
          IF(ISETKW(IENTRY).EQ.+1) 
     &    WRITE(LUOUT,'(10X,A)') KEYWOR(IENTRY)
        END DO
        WRITE(6,*)
*
        WRITE(6,*) ' ERRORS were detected for the following KEYWORDS'
        DO IENTRY = 1, MXPKW
           IF(ISETKW(IENTRY).EQ.-1) WRITE(LUOUT,'(10X,A)')KEYWOR(IENTRY)
        END DO
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
     &  '     An expert is a man who has made all the mistakes,'
        WRITE(LUOUT,*)
     &  '     which can be made, in a very narrow field        '
        WRITE(LUOUT,*)
     &  '                                                      '
        WRITE(LUOUT,*)
     &  '                                      Niels Bohr      '

        IF(IEXPERT.EQ.0) THEN
          STOP' Error in input'
        ELSE
          WRITE(6,*) ' Program continues (EXPERT mode )'
        END IF
      END IF
*
**********************************************************************
*                                                                    *
* Part 2: Insert defaults for missing optional keywords             *
*          and print error messages for missing mandatory keywords   *
*                                                                    *
**********************************************************************
*
      NMISS = 0
*
*.1: Default title
*
      IF(ISETKW(1).EQ.0) THEN
        TITLEC(1) =
     &  ' Some molecule or some atom                                  '
        TITLEC(2) =
     &  ' Some type of CI expansion                                  '
        TITLEC(3) =
     &  ' Some user who is too lazy to supply a TITLE                 '
        ISETKW(1) = 2
      END IF
*
*.2  Missing pointgroup ( has actually been defaulted )
*
      IF(ISETKW(2).EQ.0) THEN
        PNTGRP = 1
        ISETKW(2) = 2
      END IF
*
*.3 Missing number of irreps, allowed for D2h, illegal else
*
      IF(ISETKW(3).EQ.0) THEN
        IF(PNTGRP .EQ.1 ) THEN
*. Repeat default
          NIRREP = 8
          NSMCMP = NIRREP
          NSMOB  = NIRREP
          ISETKW(3) = 2
        ELSE
*. Number of irreps is mandatory for CINV,DINFH,O3
          NMISS = NMISS + 1
          WRITE(LUOUT,*)
     &    '  Input error ! .NIRREP must be specified for CinV,DinH,O3'
        END IF
      END IF
*
* 4: Internal CI expansion
*
*.Default is CAS 
      IF(ISETKW(50) .EQ. 0 ) THEN
*. Well no GASSPACES, may be a Hartree-Fock optimization..
      NGAS = 0
C     IF(ISETKW(4).EQ.0) THEN
C       INTSPC = 1
C       ISETKW(4) = 2
*. If a RAS1 or a RAS 3 space has been defined, RAS must have
*  been specified
C       IF(ISETKW(9).EQ.1.OR.ISETKW(11).EQ.1) THEN
C        ISETKW(4) = 0
C        NMISS = NMISS + 1
C         WRITE(LUOUT,*)
C    &    '  Input error ! .RAS must be specified when .RAS1 or .RAS3'
C         WRITE(LUOUT,*)
C    &    '                 has been activated '
C
C       END IF
C     END IF
C     ELSE IF (ISETKW(50) .EQ. 0 ) THEN
*. FCI expansion
C       INTSPC = 3
      END IF
*
* 6: Number of active electrons
*
*. Mandatory
      IF(ISETKW(6).EQ.0) THEN
        NMISS = NMISS + 1
          WRITE(LUOUT,*)
     &    '  Input error ! .NACTEL must be specified '
      END IF
*
* 7: Inactive orbitals
*
      IF(ISETKW(7).EQ.0) THEN
        CALL ISETVC(NINASH,0,NIRREP)
        ISETKW(7) = 0
      END IF
*
* 8: Core orbitals, only of interest if EXTSPC .ne. 0
*
      IF(ISETKW(8).EQ.0) THEN
        CALL ISETVC(NRS0SH,0,NIRREP)
        MNHR0 = 0
        IF(EXTSPC.EQ.0) THEN
          ISETKW(8) = 3
        ELSE
          ISETKW(8) = 2
        END IF
      END IF
*
* 9: RAS 1 orbitals
*
      IF(ISETKW(9).EQ.0) THEN
        CALL ISETVC(NRSSH(1,1),0,NIRREP)
        IF(INTSPC.EQ.1) THEN
          ISETKW(9) = 3
        ELSE
          ISETKW(9) = 2
        END IF
      END IF
*
* 10: RAS 2 orbitals
*
      IF(ISETKW(10).EQ.0) THEN
        CALL ISETVC(NRSSH(1,2),0,NIRREP)
        ISETKW(10) = 2
      END IF
*
* 11: RAS 3 orbitals
*
      IMLCR3 = 0
      IF(ISETKW(11).EQ.0) THEN
        CALL ISETVC(NRSSH(1,3),0,NIRREP)
        IF(MOLCS.EQ.1.AND.INTSPC.EQ.2) THEN
*. Use information from one-electron integral file to obtain
* default 
          IMLCR3 = 1
        END IF
        IF(INTSPC.EQ.1) THEN
          ISETKW(11) = 3
        ELSE
          ISETKW(11) = 2
        END IF
      END IF
*
* 12: Partitioning of secondary space ( default 1 set in SECOND)
*
C     IF(ISETKW(12).EQ.0.OR.ISETKW(12).EQ.2) THEN
C       MXR4TP = 1
C       IF(EXTSPC.EQ.0) THEN
C         ISETKW(12) = 3
C       ELSE
C         ISETKW(12) = 2
C       END IF
C     END IF
*
* 13: Secondary space
*
      IF(ISETKW(13).EQ.0) THEN
        DO IRREP = 1,  NIRREP 
          NSECSH(IRREP) = 0
        END DO
        ISETKW(13) = 2
      END IF
*
* 14: occupation restrictions for Reference space
*
      IF(ISETKW(14).EQ.0) THEN
          ISETKW(14) = 2
      END IF
*
* 15: Selection of active configurations
*
      IF(ISETKW(15).EQ.0) THEN
*. Standard is no selection
        INTSEL = 0
      END IF
*
* 16: Two times spin projection
*
      IF(ISETKW(16).EQ.0) THEN
        WRITE(LUOUT,*)
     &  '  Input error ! .MS2 must be specified '
        NMISS = NMISS + 1
      END IF
*
* 17: Spin multiplicity
*
*. Spin multiplicities: May be unspecified if NOCSF has been flagged,
*. this is tested later 
      IF(ISETKW(17).EQ.0) THEN
        ISETKW(17) = 2
        MULTSP = 0
      END IF
*
* 18: Reference symmetry
*
      IF(ISETKW(18).EQ.0) THEN
        WRITE(LUOUT,*)
     &  '  Input error ! .IREFSM must be specified '
        NMISS = NMISS + 1
      END IF
*
* 19: Roots to be optimized
*
      IF(ISETKW(19).EQ.0) THEN
        WRITE(LUOUT,*)
     &  '  Input error ! .ROOTS must be specified '
        NMISS = NMISS + 1
      END IF
*
* 20: Diagonalization routine
*
      IF(ISETKW(20).EQ.0) THEN
*. Standard is currently MICDV*
        IDIAG = 1
        ISETKW(20) = 2
      END IF
*
* 21: Explicit Hamiltonian
*
      IF(ISETKW(21).EQ.0) THEN
*. Default is no explicit Hamiltonian
        MXP1 = 0
        MXP2 = 0
        MXQ  = 0
        ISPSPC_SEL = 0
        ISETKW(21) = 2
      END IF
*
* 22: Largest allowed number of CI iterations per root
*
      IF(ISETKW(22).EQ.0) THEN
*. Default is 20 ( not active I expect )
        MAXIT = 20
        ISETKW(22) = 2
      END IF
*
* 23: Restart option
*
      IF(ISETKW(23).EQ.0) THEN
*. Default is no explicit Hamiltonian
        IRESTR = 0
        ISETKW(23) = 3
      END IF
*
* 24: Integral import
*
      IF(ISETKW(24).EQ.0) THEN
*. Default is - from NOV26: Dalton
        INTIMP = 5
        ENVIRO(1:6) = 'DALTON'
        ISETKW(24) = 2
      END IF
*
* 25: INCORE option for integrals
*
      IF(ISETKW(25).EQ.0) THEN
        IF(EXTSPC.EQ.0 ) THEN
          INCORE = 1
        ELSE
          INCORE = 0
        END IF
        ISETKW(25) = 2
C  
C       IF(INTEXP.EQ.0) THEN
C         ISETKW(25) = 3
C       ELSE
C         ISETKW(25) = 3
C       END IF

      END IF
*
* 26: DELETEd shells
*
      IF(ISETKW(26) .EQ. 0 ) THEN
*. If CAS + Active have been set or RAS + Ras3 have been set,
*. obtain for MOLCAS Interface from number of basis functions
        IF(INTSPC.EQ.1.OR.
     &    (INTSPC.EQ.2.AND.ISETKW(11).EQ.1)) THEN
          IMLCR3 = 2
        ELSE
          CALL ISETVC(NDELSH,0,NIRREP)
        END IF
        ISETKW(26) = 2
      END IF
*
* 27: Ms combinations
*
      IF(ISETKW(27).EQ.0) THEN
        PSSIGN = 0.0D0
        ISETKW(27) = 2
      ELSE IF(MS2.NE.0) THEN
        WRITE(LUOUT,*) ' Spin combinations only allowed with MS2 = 0'
        WRITE(LUOUT,*) ' Your value of MS2 = ',MS2, ' differs from zero'
        WRITE(LUOUT,*) ' LUCIA will neglect your nice suggestion '
        WRITE(LUOUT,*)  ' to use spin combinations '
        PSSIGN = 0.0D0
        ISETKW(27) = 2
      END IF
*
* 28: Ml combinations
*
      IF(ISETKW(28).EQ.0) THEN
        PLSIGN = 0.0D0
        ISETKW(28) = 2
      ELSE IF(PNTGRP.EQ.1) THEN
        WRITE(LUOUT,*) ' Ml combinations not allowed with d2h '
        WRITE(LUOUT,*) ' LUCIA will neglect your nice suggestion '
        WRITE(LUOUT,*)  ' to use ML combinations '
        PLSIGN = 0.0D0
        ISETKW(28) = 2
      ELSE IF(IREFML.NE.0) THEN
        WRITE(LUOUT,*) ' ML combinations only allowed with ML = 0'
        WRITE(LUOUT,*) 
     &  ' Your value of IREFML = ',IREFML, ' differs from zero'
        WRITE(LUOUT,*) ' LUCIA will neglect your nice suggestion '
        WRITE(LUOUT,*)  ' to use ML combinations '
        PLSIGN = 0.0D0
        ISETKW(28) = 2
      END IF
      IF(PSSIGN.EQ.0.0D0.AND.PLSIGN.EQ.0.0D0) THEN
        IDC = 1
      ELSE IF(PSSIGN.NE.0.0D0.AND.PLSIGN.EQ.0.0D0) THEN
        IDC = 2
      ELSE IF(PSSIGN.EQ.0.0D0.AND.PLSIGN.NE.0.0D0) THEN
        IDC = 3
      ELSE IF(PSSIGN.NE.0.0D0.AND.PLSIGN.NE.0.0D0) THEN
        IDC = 4
      END IF
C?    WRITE(6,* ) ' TEST readin IDC = ', IDC
*     
* 29: print flag for string information
*     
      IF(ISETKW(29).EQ.0) THEN
        IPRSTR = 0
        ISETKW(29) = 2
      END IF
*
* 30: print flag for string information
*
      IF(ISETKW(30).EQ.0) THEN
        IPRCIX = 0
        ISETKW(30) = 2
      END IF
*
* 31: print flag for orbital information
*
      IF(ISETKW(31).EQ.0) THEN
        IPRORB = 1
        ISETKW(31) = 2
      END IF
*
* 32: print flag for diagonalization information
*
      IF(ISETKW(32).EQ.0) THEN
        IPRDIA = 3
        ISETKW(32) = 2
      END IF
*
* 36: print flag for External blocks 
*
      IF(ISETKW(36).EQ.0) THEN
        IPRXT  = 0
        ISETKW(36) = 2
      END IF
*
* 43: Print occupation of lowest SD's / configurations
*
      IF(ISETKW(43).EQ.0) THEN
        IPROCC = 0
        ISETKW(43) = 2
      END IF
*
* 65: Print level for densities, default is to print  
*      natural occupation numbers only 
*
      IF(ISETKW(65).EQ.0) THEN
        IPRDEN = 1
        ISETKW(65) = 2
      END IF
*
* 84: Print level for response, default is to print  
*      final response functions as well as contributions
*
      IF(ISETKW(84).EQ.0) THEN
        IPRRSP = 3
        ISETKW(84) = 2
      END IF
*
* 99: Print level for Property: Default is to  print  
*      final values and nat.  occ decomposition
*      reduce to 1 to print only final values
*
      IF(ISETKW(99).EQ.0) THEN
        IPRPRO = 5
        ISETKW(99) = 2
      END IF
*
* 106: Print level for Coupled Cluster     
*
      IF(ISETKW(106).EQ.0) THEN
        IPRCC = 2
        ISETKW(106) = 2
      END IF
*
* 33: Number of Ci vectors in subspace
*
      IF(ISETKW(33).EQ.0) THEN
* default is 3/2 vectors per root
        IF(IDIAG.EQ.1) THEN
          MXCIV = 3 * NROOT
        ELSE 
          MXCIV = 2 * NROOT
        END IF
        ISETKW(33) = 2
      END IF
*
      MXCIV_ORIG = MXCIV
      IF(ISETKW(33).EQ.1.AND.MXCIV .LT.2*NROOT) THEN
        WRITE(LUOUT,*)
     &  '   The number of vectors is increased to 2*NROOT = ',2*NROOT
        MXCIV = 2 * NROOT
      END IF
*
      IF(IDIAG.EQ.2 .AND. MXCIV.GT.2 ) THEN
        MXCIV = 2
        NWARN = NWARN + 1
        WRITE(6,*) ' Warning: You have specified TERACI '
        WRITE(6,*) '           I allow myself to set MXCIV = 2 '
        WRITE(6,*)              
        WRITE(6,*) '                   Best Wishes    '
        WRITE(6,*) '                      Lucia       '
      END IF
     
*
* 34: CI storage mode
*
      IF(ISETKW(34).EQ.0) THEN
*. Default is three type-type-symmetry blocks
        ICISTR = 3
        ISETKW(34) = 2
      END IF
      IF(ICISTR.EQ.1.AND.ISETKW(183).EQ.1) THEN
*. Batching of CSF's in use, batch therefore also SD's
       IF(ICNFBAT.EQ.2) THEN
         ICISTR = 2
         WRITE(6,*) ' CSF batching in use '
         WRITE(6,*) ' SD batching is therefore enforced '  
         WRITE(6,*) ' ICISTR raised to 2 '
       END IF
      END IF
COLD  IF(ICISTR.EQ.1) THEN
*. complete vectors should not be used together with PICO
COLD  WRITE(LUOUT,*)
COLD &'    You have suggested the use of two complete vectors in core'
COLD  WRITE(LUOUT,*)
COLD &'    Although this could be an interesting suggestion '
COLD  WRITE(LUOUT,*)
COLD &'    I allow myself to reduce the storage mode to 3 sym. blocks '
COLD  ICISTR = 2
COLD  END IF
*
* 35: Employ CSF expansion ?
*
*. Default is no: Should be changed, but not here to 
*. ensure old input are working 
*. 
      IF(ISETKW(35).EQ.0) THEN
*. If multiplicity was defined use CSF's
        IF(ISETKW(17).EQ.1) THEN
          NOCSF = 0
        ELSE 
          NOCSF = 1
          ICNFBAT = 0
        END IF
        ISETKW(35) = 2
      END IF
      IF(ISETKW(35).EQ.1.AND.ISETKW(17).EQ.1) THEN
*. Both multiplicity and NOCSF was flagged, MULTS has priority
          WRITE(6,*) ' MULTS and NOCSF both specified'
          WRITE(6,*) ' NOCSF will be discarded '
          NOCSF = 0
          NWARN = NWARN + 1
      END IF
*
*. Note: Currently the absence of NOCSF and MULTS 
*. will result in a run without CSF's without warnings
*. should maybe be changed
*
* CSF expansion must only be used when two vectors are stored in CORE
      IF(NOCSF.EQ.0.AND.ICISTR.EQ.1.AND.ICNFBAT.EQ.2) THEN
        WRITE(LUOUT,*)
     &  ' Batched CSF calculation was specified by ICNFBAT .ge. 2 '
        WRITE(LUOUT,*)
     &  '   This requires batching of determinants '
        WRITE(LUOUT,*)
     &  '   I will set CISTOR(ICISTR) to 2 '
        ICISTR = 2
      END IF
*
* 37: Avoid any readin of integrals ( useful for obtaining
*      size of CI expansion etc.
*
      IF(ISETKW(37).EQ.0 ) THEN
        NOINT = 0
        ISETKW(37) = 2
      END IF
*
* 38: Dump integrals in formatted form: Default is No
*
      IF(ISETKW(38).EQ.0) THEN
        IDMPIN = 0
        ISETKW(38) = 2
      END IF
*. If import is from LUCIA, dumping of integrals is disabled
*. Disabling is disabled (sic): To allow for final integraltrans
C     IF(IDMPIN.EQ.1.AND.ENVIRO(1:5).EQ.'LUCIA') THEN
C       IDMPIN = 0
C       WRITE(6,*) 'Warning: Dump to LUCIA format disabled'
C       WRITE(6,*) '(input format is LUCIA !              )'
C       WRITE(6,*)
C       WRITE(6,*) '                     /Lucia            '
C     END IF

*
* 39: Explicitly dimension of dimension of block of resolution strings
*
      IF(ISETKW(39).EQ.0) THEN
        MXINKA = 100
        ISETKW(39) = 2
      END IF
*
* 40: Use CJKAKB intermediate matrices in alpha-beta loop,
*      Default is  YES !!!!!
*
      IF(ISETKW(40).EQ.0) THEN
        ICJKAIB = 1
        ISETKW(40) = 2
      END IF
*
*  41: Initial CI in reference space, default is: No
*
      IF(ISETKW(41).EQ.0) THEN
         ISETKW(41) = 2
         INIREF = 0
      END IF
*
*  42: Restart with CI in reference space             
*
      IF(ISETKW(42).EQ.0) THEN
         ISETKW(42) = 2
         IRESTRF = 0
      END IF
*
*  44: Use MOC method for alpha-alpha loop, default is NO !
*
      IF(ISETKW(44).EQ.0) THEN
         ISETKW(44) = 2
         MOCAA = 0
      END IF
*
*  45: Use MOC method for alpha-beta loop, default is NO !
*
      IF(ISETKW(45).EQ.0) THEN
         ISETKW(45) = 2
         MOCAB = 0
      END IF
*
* Core energy: Default is 0 / MOLCAS: Value read in !
*
      IF(ISETKW(46).EQ.0) THEN
         ISETKW(46) = 2
         ECORE = 0.0D0
      END IF
*
*. Use perturbation theory for zero order space . def is no !
*
      IF(ISETKW(47).EQ.0) THEN
        IPERT = 0
        NPERT = 0
        ISETKW(47) = 2 
*. Else ensure that a CI in reference space is performed
      ELSE
        INIREF = 1
      END IF
*
*
*. 48: Approximate Hamiltonian in reference space: NO !!
*
      IF(ISETKW(48).EQ.0) THEN
        IAPRREF = 0
        MNRS1RE = MNRS1R
        MXRS3RE = MXRS3R
        ISETKW(48) = 2 
      END IF
*
*. 49: Approximate Hamiltonian in zero order space: NO !!
*
      IF(ISETKW(49).EQ.0) THEN
        IAPRZER = 0
        MNRS1ZE = MNRS10 
        MXRS3ZE = MXRS30 
        ISETKW(49) = 2 
      END IF
*
* 50: GAS shells must be defined 
*
      IF(ISETKW(50).EQ.0) THEN
        WRITE(6,*) ' GASSH must be defined '
        NERROR = NERROR + 1
        IGSFILL = 0
        ISETKW(50) = -1
      END IF
*
* 52: Combination of gasspaces: Default is just to take each  space
*      By itself
*
      IF(ISETKW(52).EQ.0) THEN
        NCMBSPC = NCISPC
        DO ICISPC = 1, NCISPC
          LCMBSPC(ICISPC) = 1
          ICMBSPC(1,ICISPC) = ICISPC
        END DO
        ISETKW(52) = 2
      END IF
*
* 53: Convergence threshold for CI                                  
*
      IF(ISETKW(53).EQ.0) THEN
        THRES_E = 1.0D-10
        ISETKW(53) = 2
      END IF
*
* 54: General sequencer: default is just straight sequence 
*      of CI with default number of iterations
      IF(ISETKW(54).EQ.0) THEN
        DO JCMBSPC = 1, NCMBSPC
          NSEQCI(JCMBSPC) = 1
          CSEQCI(1,JCMBSPC) = 'CI'
          ISEQCI(1,JCMBSPC) = MAXIT 
        END DO
        ISETKW(54) = 2
      END IF
*
* 55: EKT calculation: Default is no
*
      IF(ISETKW(55).EQ.0) THEN
        IEXTKOP = 0
        ISETKW(55) = 2
      END IF
*
*. 56: Default Machine: Good old BIM machine
*
      IF(ISETKW(56).EQ.0) THEN
        MACHINE(1:3) = 'IBM'
        ISETKW(56) = 2
      END IF
*
* 57: Allow first order correction to be saved on DISC
*     (For vector free calculations )
*     Default is: NO !!
      IF(ISETKW(57).EQ.0) THEN
        IC1DSC = 0
        ISETKW(57) = 2
      END IF
*
* 58: Restrictions on interactions of perturbation
*
*. Default is: no 
      IF(ISETKW(58).EQ.0) THEN
        NPTSPC = 0
        IH0SPC = 0
        ISETKW(58) = 2
      END IF
*
* 59: Type of perturbation in subspaces spaces
*
* Default is specified by IPART from keyword PERTU
      IF(ISETKW(59).EQ.0) THEN
       ISETKW(59) = 2
       IF(IH0SPC.NE.0) THEN
         DO JPTSPC = 1, NPTSPC
           IH0INSPC(JPTSPC) = IPART
         END DO
       END IF
      END IF
*
* 60: Reference Root, default is NROOT
*
*. Should be less or equal to NROOT
      IF(ISETKW(60).EQ.1) THEN
        IF(IRFROOT.GT.NROOT) THEN
          WRITE(6,*) ' Reference root (RFROOT) larger '
          WRITE(6,*) ' than total number of roots (NROOT) '
          WRITE(6,*) ' CHANGE NROOT or RFROOT '
          NMISS = NMISS + 1
        END IF
      END IF

      IF(ISETKW(60).EQ.0) THEN
       ISETKW(60) = 2
       IRFROOT = NROOT
      END IF
*
* 61: H0EX: Orbital spaces in which exaxt Hamiltonian is used
*      No default
*.
*. Is H0EX required ( Has H0FRM = 5 been used )
      IUSED = 0
      IF(ISETKW(59).EQ.1) THEN
         IUSED = 0
         DO JPTSPC = 1, NPTSPC
           IF( IH0INSPC(JPTSPC) .EQ. 5 ) IUSED = 1
         END DO
       END IF
       IF(IUSED.EQ.0.AND.ISETKW(61).EQ.0) THEN
*. No exact spaces included and none have been defined !
         NH0EXSPC = 0
         IH0EXSPC(1) = -1
       END IF
       IF(IUSED.EQ.1.AND.ISETKW(61).EQ.0) THEN
*. Needed, but not supplied
          WRITE(6,*) ' You have specified that zero order operator'
          WRITE(6,*) ' Include exact Hamilton operator in subspace'
          WRITE(6,*) ' You should then also supply Keyword H0EX '
          NMISS = NMISS + 1
       END IF
*
*. If perturbation theory will be invoked be sure that the 
*. form of perturbation theory has been specified through 
* KEYWORD PERTU ( number 47 as you maybe know )
      IDOPERT = 0
      DO JCMBSPC = 1, NCMBSPC
        DO JSEQCI = 1, NSEQCI(JCMBSPC)
          IF(ISEQCI(JSEQCI,JCMBSPC).EQ.-5 ) IDOPERT = 1
        END DO
      END DO
*
      IF(IDOPERT.EQ.1 .AND. IPERT.EQ.0) THEN
        WRITE(6,*) ' Perturbation theory will be used '
        WRITE(6,*) ' Please specify form through PERTU keyword '
        NMISS = NMISS + 1   
      END IF
*
*. 62: Default Handling of degenrences of initial CI vectors
*.      Default is: No action
*
      IF(ISETKW(62).EQ.0) THEN
        INIDEG = 0
        ISETKW(62) = 2
      END IF
*
*. 63: Use F + Lambda(H-F) as operator instead of H          
*.      Default is: No i.e Lambda = 1
*
      IF(ISETKW(63).EQ.0) THEN
        XLAMBDA = 1.0D0
        ISETKW(63) = 2
      END IF
*
*. 64: Smallest block in batch of C and sigma                
*.      Default is zero                
*
      IF(ISETKW(64).EQ.0) THEN
        LCSBLK = 0       
        ISETKW(64) = 2
      END IF
*
*. 66: NO MO file: Default is access to MO-AO file          
*
      IF(ISETKW(66).EQ.0) THEN
        NOMOFL = 0
        ISETKW(66) = 2
      END IF
*
*. 68: Type of Final orbitals, default is no construction
*
      IF(ISETKW(68).EQ.0) THEN
        ISETKW(68) = 2
        IFINMO = 0
      END IF
*
*. 69: Default Threshold for individual energy correction = 0.0            
*
      IF(ISETKW(69).EQ.0) THEN
        E_THRE = 0.0D0
        ISETKW(69) = 2
      END IF
*
*. 70: Default Threshold for wave individual function corrections = 0.0 
*
      IF(ISETKW(70).EQ.0) THEN
        C_THRE = 0.0D0
        ISETKW(70) = 2
      END IF
*
*. 71: Default Threshold for total energy corrections = 0.0 
*
      IF(ISETKW(71).EQ.0) THEN
        E_CONV = 0.0D0
        ISETKW(71) = 2
      END IF
*
*. 72: Default Threshold for total wave function correction = 0.0            
*
      IF(ISETKW(72).EQ.0) THEN
        C_CONV = 0.0D0
        ISETKW(72) = 2
      END IF
*
*. 73: Perform Class selection: Default if Yes if TERACI is used      
*
      IF(ISETKW(73).EQ.0) THEN
        IF(IDIAG.EQ.1) THEN  
          ICLSSEL = 0     
        ELSE IF (IDIAG.EQ.2) THEN
          ICLSSEL = 1
        END IF
        ISETKW(73) = 2
      END IF
*
* 74: Calculation of density matrices: Default is 
*       calculation of one-body density
*       but not for CC since this is not completely trivial
*
      IF(ISETKW(74).EQ.0) THEN
        IDENSI = 1
        IF(I_DO_CC.EQ.1) THEN
          IDENSI = 0
          WRITE(6,*) ' 1-el. density default disabled for CC'
        END IF
        ISETKW(74) = 2
      END IF
*. If IDENSI was set to zero and properties were requested
*  overwrite input to obtain 1-el matrix
      IF(IDENSI.EQ.0.AND.ISETKW(80).EQ.1) THEN
        WRITE(6,*) ' You have specified calculation of'
        WRITE(6,*) ' one-electron properties, and this'
        WRITE(6,*) ' requires the calculation of the '
        WRITE(6,*) ' one-electron density. '
        WRITE(6,*)
        WRITE(6,*) ' You have, however, specified IDENSI=0'
        WRITE(6,*) ' corresponding  to no densities'
        WRITE(6,*)
        WRITE(6,*) ' I will allow myself to modify your'
        WRITE(6,*) ' input to allow calculation of the '
        WRITE(6,*) ' one-electron densities, so property'
        WRITE(6,*) ' calculation can proceed as planned '
        WRITE(6,*)
        WRITE(6,*)                        ' Lucia '
*. and do it
        IDENSI = 1
      END IF
*. If CC is performed, one- and two- particle densities are
*  used in current simple-minded implementation. 
COLD  IF(I_DO_CC .EQ. 1 .AND. IDENSI.LE.1 ) THEN
COLD    IDENSI = 2
COLD    WRITE(6,*) ' IDENSI flag raised to two for CC calculation'
COLD  END IF
*. Two-electron density also needed for MCSCF
      IF((I_DO_MCSCF.EQ.1.OR. I_DO_NORTMCSCF.EQ.1)
     &   .AND. IDENSI.LE. 1) THEN
        IDENSI = 2
C       WRITE(6,*) ' IDENSI flag raised to 2 for MCSCF calculation'
      END IF
*
* If spindensities have been requested, calculate also 
* the corresponding densities 
      IF(ISPNDEN.GT.IDENSI) THEN
        IDENSI = ISPNDEN
        WRITE(6,*) ' DENSI keyword raised to SPNDEN keyword '
      END IF
*
* 75: Perturbation expansion of EKT, default is no 
*
      IF(ISETKW(75).EQ.0) THEN
        IPTEKT = 0
        ISETKW(75) = 2
      END IF
*
* 76: Root for zero order operator , default is NROOT
*
*. Should be less or equal to NROOT
      IF(ISETKW(76).EQ.1) THEN
        IF(IH0ROOT.GT.NROOT) THEN
          WRITE(6,*) ' Zero order operator root (H0ROOT) larger '
          WRITE(6,*) ' than total number of roots (NROOT) '
          WRITE(6,*) ' CHANGE NROOT or H0ROOT '
          NMISS = NMISS + 1
        END IF
      END IF
      IF(ISETKW(76).EQ.0) THEN
       ISETKW(76) = 2
       IH0ROOT = NROOT
      END IF
*
* 77: NO restart from previous vectors in calculation 2
*      Deafault is NO NO, ie. restart in calc 2
*
      IF(ISETKW(77).EQ.0) THEN
        IRST2 = 1
        ISETKW(77) = 2
      END IF
*
* 78: skip initial energy evaluations - if possible
*
      IF(ISETKW(78).EQ.0) THEN
        ISKIPEI = 1
        ISETKW(78) = 2
      END IF
* 
* 79: Symmetry of x,y,z - needed for property calculations
*
      IF(ISETKW(79).EQ.0) THEN
*. Problematic if Properties should be calculated
       IF(ISETKW(80).EQ.1.OR.ISETKW(81).EQ.1.OR.ISETKW(82).EQ.1)
     & THEN
         WRITE(6,*) ' Symmetry of X,Y,Z has not been given'
         WRITE(6,*) ' You have to specify this for property calc'
         WRITE(6,*) ' Please add KEYWORD XYZSYM '
         NMISS = NMISS + 1
         ISETKW(79) = -1
       ELSE
*. Is not needed, just supply zeroes
         DO ICOMP = 1, 3
           IXYZSYM(ICOMP) = 0
         END DO
         ISETKW(79) = 2
       END IF
      END IF
*
* 80: Property calculation, default is no
*
      IF(ISETKW(80).EQ.0) THEN
        NPROP = 0
        ISETKW(80) = 2
      END IF
*
* 81: Transition properties , default is no
*
      IF(ISETKW(81).EQ.0) THEN
        ITRAPRP = 0
        ISETKW(81) = 2
      END IF
*
* 82: Response properties , default is no
*
      IF(ISETKW(82).EQ.0) THEN
        IRESPONS = 0
        NRSPST = 0
        ISETKW(82) = 2
        NRESP = 0
        N_AVE_OP = 0
      END IF
*. Properties should be defined if transition properties are
*. invoked
      IF(ITRAPRP.NE.0.AND.NPROP.EQ.0) THEN
        WRITE(6,*) 
     &  ' You have specified transition property calculation'
        WRITE(6,*)
     &  ' (keyword TRAPRP) but no property labels have been supplied'
        WRITE(6,*)
     &  '(Keyword PROPER). Transition densities will be obtained '
      END IF
*
* 83: Max number of iterations in linear equations
*
      IF(ISETKW(83).EQ.0) THEN
        MXITLE = 30
        ISETKW(83) = 2
      END IF
*
* 85: Root homing, default is no                    
*
      IF(ISETKW(85).EQ.0) THEN
        IROOTHOMING = 0
        ISETKW(85) = 2
      END IF
*
* 86: Particle hole simplifications, default is no
*
      IF(ISETKW(86).EQ.0) THEN
       IUSE_PH = 0
       ISETKW(86) = 2
      END IF
*
* 87: Ask advice for route in sigma blocks, default is no
*      (It is said that programs reflects the minds of their creators)
*
      IF(ISETKW(87).EQ.0) THEN
       IADVICE = 0
       ISETKW(87) = 2
      END IF
*
* 88: Transform CI vectors at end of each calculation    
*      default is no
*
      IF(ISETKW(88).EQ.0) THEN
       ITRACI = 0
       ISETKW(88) = 2
       ITRACI_CR = 'undefine'
       ITRACI_CN = 'undefine'
      END IF
*
* 89: Divide strings into active and passive parts
*      default is no
*
      IF(ISETKW(89).EQ.0) THEN
       IUSE_PA = 0
       ISETKW(89) = 2
      END IF
*
* 90: Perturbation expansion of Fock matrix: default is no
*
      IF(ISETKW(90).EQ.0) THEN
       IPTFOCK = 0
       ISETKW(90) = 2
      END IF
*
* 91: Print final CI vectors: default is no
*
      IF(ISETKW(91).EQ.0) THEN
       IPRNCIV = 0
       ISETKW(91) = 2
      END IF
*
* 92: Restart CC calculation with coefs on LU_CCAMP
*
      IF(ISETKW(92).EQ.0) THEN
       I_RESTRT_CC = 0
       ISETKW(92) = 2
      END IF
*
* 93: End Calculation with integral transformation 
*
      IF(ISETKW(93).EQ.0) THEN
       ITRA_FI = 0     
       ISETKW(93) = 2
      END IF
*. Requires access to MO-AO file
      IF(ITRA_FI.EQ.1) THEN
       IF(NOMOFL.EQ.1) THEN
         WRITE(6,*) ' Integral transformation required, '
         WRITE(6,*) ' but no mo-ao file accessible      '
         WRITE(6,*) ' MO-MO integral transformation '        
C        WRITE(6,*) ' REMOVE KEWORD NOMOFL '
C        ISETKW(93) = -1
C        NERROR = NERROR + 1
       END IF
*. Integrals will be written in LUCIA format, so set IDMPIN flag
       IDMPIN = 1
C?     WRITE(6,*) ' DMPINT flag set to one '
      END IF
*
* 94: Initialize Calculation with integral transformation 
*
      IF(ISETKW(94).EQ.0) THEN
       ITRA_IN = 0     
       ISETKW(94) = 2
      END IF
*. Requires access to MO-AO file
       IF(ITRA_IN.EQ.1.AND.NOMOFL.EQ.1) THEN
         WRITE(6,*) ' Integral transformation required, '
         WRITE(6,*) ' but no mo-ao file accessible      '
         WRITE(6,*) ' REMOVE KEWORD NOMOFL '
         ISETKW(94) = -1
         NERROR = NERROR + 1
       END IF
*
* 95: Multispace optimization in each run, default is no
*
      IF(ISETKW(95).EQ.0) THEN
        MULSPC = 0
        IFMULSPC = 0
        LPAT = 0
        ISETKW(95) = 2
      END IF
*
* Use relaxed densities for properties: default is no
*
      IF(ISETKW(96).EQ.0) THEN
        IRELAX = 0
        ISETKW(96) = 2
      END IF
*.
      IF(IRELAX.EQ.1) THEN
*. To obtain relaxed densities two-elec density must be calc, so
        IF(IDENSI.LT.2) THEN
          WRITE(6,*) ' Density matrix flag (IDENSI) raised '
          WRITE(6,*) ' to allow calculation of 2-elec densities'
          IDENSI = 2
        END IF
      END IF
*
* Expert mode ( neglect mistyped keywords ): default is no expert
*
      IF(ISETKW(97).EQ.0) THEN
        IEXPERT = 0
        ISETKW(97) = 2
      END IF
*
* Number of roots to be converged: default is total number of roots
*
      IF(ISETKW(98).EQ.0) THEN
        NCNV_RT = NROOT
        ISETKW(98) = 2
      END IF
*
* 100: Do quantum dot calculation, default is no
*
      IF(ISETKW(100).EQ.0) THEN
        IDOQD = 0
        ISETKW(100) = 2
      END IF
*
* 101: Restrict MS2 at some intermediate level: default is no way
*
      IF(ISETKW(101).EQ.0) THEN
        I_RE_MS2_SPACE = 0
        I_RE_MS2_VALUE = 0
        ISETKW(101) = 2
      END IF
*
* 102: Form of preconditioner, default is sd diagonal                
*
      IF(ISETKW(102).EQ.0) THEN
        IPRECOND = 1 
        ISETKW(102) = 2
      END IF
*
* 103: Treat all TT blocks with given types simultaneously : Default is no
*
      IF(ISETKW(103).EQ.0) THEN
        ISIMSYM = 0
        ISETKW(103) = 2
      END IF
*
* 104: Use hardwired loops for selected terms: default is no
*
      IF(ISETKW(104).EQ.0) THEN
        IUSE_HW = 0
        ISETKW(104) = 2
      END IF
*
* 105: Use Full H0 including projection operators for Lambda calculations
*
      IF(ISETKW(105).EQ.0) THEN
        IUSEH0P = 0
        ISETKW(105) = 2
      END IF
*
* 107: Calculate expectation value of Lz^2: Default is no
*
      IF(ISETKW(107).EQ.0) THEN
        I_DO_LZ2 = 0
        ISETKW(107) = 2
      END IF
*. If LZ2 is to be calculated the symmetry of XYZ must have been 
*. specified
      IF(I_DO_LZ2.EQ.1.AND.ISETKW(79).EQ.2) THEN
        WRITE(6,*) ' For calculating LZ2, please supply XYZSYM'
        NERROR = NERROR + 1
      END IF
*. If Lz2 is to be calculated, two-electron densities must be calculated- no, CI is used
COLD  IF(I_DO_LZ2.EQ.1.AND.IDENSI.NE.2) THEN
COLD    IDENSI = 2
COLD    WRITE(6,*) ' IDENSI has been raised by Lz flag '
COLD  END IF
*
* 108: Method used for solving CC equations, default is DIIS
*
      IF(ISETKW(108).EQ.0) THEN
c set default method: DIIS with 8 vectors
c variational method? we assume no (ivar=0), has to be corrected later
        ivar = 0               
        iorder = 1
        iprecnd = 1
        isubsp = 2
        ilsrch = 0
        icnjgrd = 0
        mxsp_sbspja = 0
        isbspjatyp = 0
        isbspja_start = 2       ! lowest possible iteration is 2
        thr_sbspja = 1d-1
        mxsp_diis = 8
        idiistyp = 2
        idiis_start = 0
        thr_diis = 1d-1
c trust radius: not active
        trini = 2.d0
        trmin = 0.25d0
        trmax = 2.0d0
        trthr1l = 0.8d0
        trthr1u = 1.2d0
        trthrfac1 = 1.2d0
        trthr2l = 0.4d0
        trthr2u = 1.6d0
        trfac1  = 1.2d0
        trfac2  = 0.8d0
        trfac3  = 0.3d0
c old:
        ICCSOLVE = 1
        ISETKW(108) = 2
      END IF
*
* 109: Setup CCN Jacobian: Default is no
*
      IF(ISETKW(109).EQ.0) THEN
        I_DO_CCN = 0
        ISETKW(109) = 2
      END IF
*
* 110: Use subspace Jacobian to improve correction vectors
*
      IF(ISETKW(110).EQ.0) THEN
        I_DO_SBSPJA = 0
        ISETKW(110) = 2
      END IF
*
* 111: Convergence Threshold for norm of residual for coupled cluster calcs
*
      IF(ISETKW(111).EQ.0) THEN
c old:
        CCCONV = 1.0D-6
        thrstp  = 1d-6
        thrgrd  = 1d-6
        thr_de  = 1d-7

        ISETKW(111) = 2
      END IF
*
* 112: Number of hole spaces in QDOT calculations, must be specified
*
      IF(ISETKW(112).EQ.0) THEN
        IF(IDOQD.EQ.1) THEN
          WRITE(6,*) ' Number of hole-orbital spaces must be specified'
          WRITE(6,*) ' (QDOT calculation) '
          ISETKW(112) = -1
          NERROR = NERROR + 1
          NMISS = NMISS + 1
        ELSE
*. NOT QDOT calculation, just set to zero
           N_HOLE_ORBSPACE = 0
           ISETKW(112) = 2
        END IF
      END IF
*
* 113: Use CC3 approximation for wave function and set up CC3 Jacobian
*
      IF(ISETKW(113).EQ.0) THEN
        I_DO_CC3 = 0
        ISETKW(113) = 2
      END IF
*
* 114: Start CC by reformatting CI coefficients 
*
      IF(ISETKW(114).EQ.0) THEN
        I_DO_CI_TO_CC = 0
        ISETKW(114) = 2
      END IF
*
* 115: Form of CC equations, default is traditional CC 
*
      IF(ISETKW(115).EQ.0) THEN
        CCFORM(1:3) = 'TCC' 
        ISETKW(115) = 2
      END IF
*
* 116: Calculate CC excitation energies after CC: Default is no
*
      IF(ISETKW(116).EQ.0) THEN
        I_DO_CC_EXC_E  = 0
        ISETKW(116) = 2
      END IF
*
* 117: Restart CC excitation energies: Default is no
*
      IF(ISETKW(117).EQ.0) THEN
        IRES_EXC = 0             
        ISETKW(117) = 2
      END IF
*
*
* 118: Dimension of resolutions strings for CC
*
      IF(ISETKW(118).EQ.0) THEN
        MXINKA_CC = 100
        LCCB = MXINKA_CC
        ISETKW(118) = 2
      END IF
*
* 119: Use spincombination for CC expansion, default is p.t. no 
*
      IF(ISETKW(119).EQ.0) THEN
         MSCOMB_CC = 0
         ISETKW(119) = 2
      END IF
*
* 120: Use similarity transformed Hamiltonian to include singles in CC
*       default is no pt
*
      IF(ISETKW(120).EQ.0) THEN
        ISIMTRH = 0
        ISETKW(120) = 2
      END IF
*
* 121: Freeze selected excitation levels in CC: Default is no
*
      IF(ISETKW(121).EQ.0) THEN
        IFRZ_CC = 0
        NFRZ_CC = 0
        ISETKW(121) = 2
      END IF
*
* 122: Calculate expectation value of H in actual space 
*
      IF(ISETKW(122).EQ.0) THEN
        I_DO_CC_EXP = 0
        ISETKW(122) = 2
      END IF
*
* 123: Form of CC vector function in use: Default is H_EF approach
*       I_DO_NEWCCV = 0 => Original codes with erroneous scaling
*       I_DO_NEWCCV = 1 => First set of codes with correct scalin (H_EF approach)
*       I_DO_NEWCCV = 2 => New commutator based coded 
*
*.  
*. VCC needs old convention
      IF(ISETKW(115).EQ.1.AND.CCFORM(1:3).EQ.'VCC') THEN
c        IF(ISETKW(123).EQ.1.OR.ISETKW(124).EQ.1) THEN
        IF(ISETKW(123).EQ.1) THEN
          WRITE(6,*) 'NEWCCV and NEWCCP are not compatible with VCC'
          WRITE(6,*) 'Remove these keywords and I will be fine!'
          IF(ISETKW(123).EQ.1) ISETKW(123)=-1
          IF(ISETKW(124).EQ.1) ISETKW(124)=-1
        ELSE
          I_DO_NEWCCV = 0
          ISIMTRH = 0
          ISETKW(123)=2
          ISETKW(124)=2
          ISETKW(136)=2
        END IF
      END IF
      IF(ISETKW(136).EQ.0.AND.ISETKW(123).EQ.0) THEN
        I_DO_NEWCCV = 1
        ISETKW(123) = 2
        ISETKW(136)=2 
      END IF
      IF(I_DO_NEWCCV.GE.1.AND.I_DO_CC.EQ.1) THEN 
*. Enforce use of simtrh to obtain singles contributions
        ISIMTRH = 1
*. Spin-combinations are not fully implemented in the 
*. new codes, ( used for computations not storing), so
*. turn on IUSE_TR
        IF(MSCOMB_CC.EQ.1) THEN
          IUSE_TR = 1
          MSCOMB_CC = 0
        END IF
      END IF
*. new ccv requires division of orbitals into holes and particles so
      IF(I_DO_CC.EQ.1.AND.I_DO_NEWCCV.GE.1.AND.IUSE_PH.EQ.0) THEN 
        IUSE_PH = 1
        WRITE(6,*) ' NEWCCV keyword enforces IUSE_PH = 1 '
      END IF
*
* 124: Use New phase for CC operators: Default is now yes
*       (Only relevant for old routines, for new routines
*       the new convention is built in)
*
      IF(ISETKW(124).EQ.0) THEN
        I_USE_NEWCCP = 1
C?      WRITE(6,*) ' New phase convention is used for CC'
        ISETKW(124) = 2
      END IF
*
* 125: Impose a limit on the allowed rank of spin-orbital 
*       excitation level. Default is now (NOV. 2003) to 
*       set max. spinorbital excitation level to max. orbital.
*       excitation level. This is communicated by setting the 
*       parameter MXSPOX to 0 
      IF(ISETKW(125).EQ.0) THEN
         MXSPOX = 0
         ISETKW(125) = 2
      END IF
*
* 126: Define mask determinant for division of spinorbitals into 
*       holes and particle
      IF(ISETKW(126).EQ.0) THEN
        I_DO_MASK_CC = 0
        ISETKW(126) = 2
      END IF
*
* 127: Eliminate rotations only in internal space: default is NO
      IF(ISETKW(127).EQ.0) THEN
        NOAAEX = 0
        ISETKW(127) = 2
      END IF
*
* 128: Spin restricted calculations: Default is no 
*
      IF(ISETKW(128).EQ.0) THEN
        ISPIN_RESTRICTED = 0
        ISETKW(128) = 2
      END IF
*. If spin-restricted calculations are prescribed, the multiplicity 
*. should be specified
      IF(ISETKW(128).EQ.1.AND.ISETKW(17).EQ.0) THEN
        WRITE(6,*) ' Spinrestricted calculation specified (SPINRS) '
        WRITE(6,*) ' But: no multiplicity specified (MULTS) ' 
        WRITE(6,*) ' Please add MULTS or remove SPINRS '
        NMISS = NMISS + 1
      END IF
*
* 129: General transition density, default is no WAY
*
      IF(ISETKW(129).EQ.0) THEN
        IGENTRD = 0
        ISETKW(129) = 2
      END IF
*
* 130: Reorder orbitals: Default is no
*
      IF(ISETKW(130).EQ.0) THEN
        I_DO_REO_ORB = 0
        NSWITCH = 0
        ISETKW(130) = 2
      END IF
*
* 131: Information on internal contraction excitation operators 
*       Set all operators to zero 
*
      IF(ISETKW(131).EQ.0.AND.ISETKW(167).EQ.0) THEN
        ICEXC_RANK_MIN = 0
        ICEXC_RANK_MAX = 0
        ICEXC_INT = 0
        I_HAVE_ICEXC_INFO = 0
      END IF
*
* 132: Comparison of CC and CI in last CC calc, default is no
*
      IF(ISETKW(132).EQ.0) THEN
        I_DO_CMPCCI = 0
        ISETKW(132) = 2
      END IF
*
* 133: Expand CC to CI expansion after last CC calc, default is no.
      IF(ISETKW(133).EQ.0) THEN
        I_DO_CC_TO_CI = 0
        ISETKW(133) = 2 
      END IF
*
*. 134: Construct complete Hamiltonian matrix. Dafault is no
*        as this became rather unfashionable three decades ago
      IF(ISETKW(134).EQ.0) THEN
        I_DO_COMHAM = 0
        ISETKW(134) = 2
      END IF
*
*. 135: Dump H-matrices in form readable for initial MRPT program , Def is no
*
       IF(ISETKW(135).EQ.0) THEN
        I_DO_DUMP_FOR_MRPT = 0
        ISETKW(135) = 2
       ELSE 
*. Will H will be dumped made sure that refspace has been defined and
*  COMHAM has been set 
        IF(ISETKW(14).NE.1) THEN
           WRITE(6,*) 
     &    ' Error: Dump of H0 and V-matrices requested(Keyword DMPMRP)'
           WRITE(6,*) ' But P-space is not defined ( use REFSPC ) '
           NMISS = NMISS + 1
           ISETKW(14) = -1
        END IF
        IF(ISETKW(134).EQ.0) THEN
*. Activate comham
          I_DO_COMHAM = 1
          WRITE(6,*) ' DMPMRP keywords activated COMHAM '
        END IF
       END IF
*
* 136: Use very new CC codes (2001-2003)
*
       IF(ISETKW(136).EQ.0) THEN
         I_DO_VERY_NEW_CC = 0
         ISETKW(136) = 2
       END IF

*. Enforce use of simtrh to obtain singles contributions
       IF(ISETKW(136).EQ.1) THEN
         ISIMTRH = 1
         IUSE_PH = 1
         IF(MSCOMB_CC.EQ.1) THEN
c           IUSE_TR = 1
           WRITE(6,*) 'CMB_CC deactivated for new CC-vectorfunction'
           MSCOMB_CC = 0
         END IF
       END IF
*
* 138: Initial guess to MO's, default is diagonalization of H1
      IF(ISETKW(138).EQ.0) THEN
         INI_HF_MO = 1
         ISETKW(138) = 2
       END IF
*
* 139: Optimization procedure for HF: Default is simple Roothaan-Hall
*
      IF(ISETKW(139).EQ.0) THEN
        IHFSOLVE = 1
        ISETKW(139) = 2
      END IF
*
* 140: Spin-densities: default is none
*
      IF(ISETKW(140).EQ.0) THEN
*. (Was actually given as default at start of this routine
*   but it is better to make the same def twice not forget it ..
        ISPNDEN = 0
        ISETKW(140) = 2
      END IF
* 141: Specification of general two-body operators
      IF (ISETKW(141).EQ.0) THEN
        INC_SING(1:3) = 0
        INC_DOUB(1:5) = 1
        IGTBCS = 0
        IGTBMOD=0
        ISYMMET_G = 0
        ISETKW(141)=2
      END IF
* 141: Unused
c      IF (ISETKW(141).EQ.0) THEN
c        ISETKW(141)=2
c      END IF
* 142: Specification of general two-body operators
      IF (ISETKW(142).EQ.0) THEN
        N_EAG = 0
        ISETKW(142)=2
      END IF
* 143: Specification of general two-body operators
      IF (ISETKW(143).EQ.0) THEN
        N_FOO = 0
        ISETKW(143)=2
      END IF
* 144: Specification of general two-body operators
      IF (ISETKW(144).EQ.0) THEN
        N_HSS = 0
        ISETKW(144)=2
      END IF
* 145: Specification of core memory
      IF (ISETKW(145).EQ.0) THEN
        MAXMEM = MXPWRD
        ISETKW(145)=2
      END IF
* 146: Specification of memory saving version
      IF (ISETKW(146).EQ.0) THEN
        ISVMEM = 0 ! default is: business as usual
        ISETKW(146)=2
      END IF
* 147: Specification of target wave-function model
      IF (ISETKW(147).EQ.0) THEN
        ITGSPC = 0  ! default is: no target model
        ITGCLC = 0
        ISETKW(147)=2
      END IF
*. 148: Calculate cumulants: Default is still no ( Sorry Werner)
      IF(ISETKW(148).EQ.0) THEN
        ISETKW(148) = 2
        ICUMULA = 0
      END IF
*. 149: Restart IC calculation, Default is no 
       IF(ISETKW(149).EQ.0) THEN
        IRESTRT_IC = 0
        ISETKW(149) = 2
       END IF
*. 150: Number of commutators employed in various parts of MRCC
       IF(ISETKW(150).EQ.0) THEN
*. Defaults corresponds to simple Jacobian approximation,
*  and no approximations for energy and vectorfunction for MRCCSD
        NCOMMU_E = 4
        NCOMMU_J = 1
        NCOMMU_V = 8
        ISETKW(150) = 2
       END IF
*
*.151:  Use approximate Hamiltonian in highest commutator for energy-function, default is zero
*
        IF(ISETKW(151).EQ.0) THEN
          I_APPROX_HCOM_E = 0
          ISETKW(151) = 2
        END IF
*
*.152:  Use approximate Hamiltonian in highest commutator for energy-function, default is zero
*
        IF(ISETKW(152).EQ.0) THEN
          I_APPROX_HCOM_V = 0
          ISETKW(152) = 2
        END IF
*
*.153:  Use approximate Hamiltonian in highest commutator for energy-function, default is zero
*
        IF(ISETKW(153).EQ.0) THEN
          I_APPROX_HCOM_J = 0
          ISETKW(153) = 2
        END IF
*
*.154:  Define orbital spaces in which density should be calculated in 
*
       IF(ISETKW(154).EQ.0) THEN
*. Default is pt all spaces 
         IDENS_IN = 1
         IDENS_AC = 1
         IDENS_SEC = 1
         ISETKW(154) = 2
       END IF
*
*. 155: Read in S and J for MRCC calculations, Default is no
*
       IF(ISETKW(155).EQ.0) THEN
          IREADSJ = 0
          ISETKW(155) = 2
       END IF
*
*. 156: Product expansions of wavefunction, default is still no
*
      IF(ISETKW(156).EQ.0) THEN
        I_DO_PRODEXP = 0
        ISETKW(156) = 2
      END IF
*
*. 157: Subspace wavefunctions included
*
      IF(I_DO_PRODEXP.EQ.1.AND.ISETKW(157).EQ.0) THEN
       WRITE(6,*) ' No subspace wave functions included '
       WRITE(6,*) ' This is required - no default pt '
       NERROR = NERROR + 1
       ISETKW(157) = -1
      END IF
*
*. 158: Max intersubspace excitation level for product wave functions, default is 0
*
      IF(I_DO_PRODEXP.EQ.1.AND.ISETKW(158).EQ.0) THEN
       INTRA_EXC_PRWF = 0
       ISETKW(158) = 2
      END IF
*
*. 159: Printflag for CSF information, default is zero
*
      IF(ISETKW(159).EQ.0) THEN
        IPRCSF = 0
        ISETKW(159) = 2
      END IF
*. 160: Include active-active excitation(without external part)
*.      Default is no way..
        IF(ISETKW(160).EQ.0) THEN
          I_INC_AA = 0
          ISETKW(160) = 2
        END IF
*
* 161 Threshold for singularities
*
      IF(ISETKW(161).EQ.0) THEN
        THRES_SINGU = 1.0D-6
        ISETKW(161) = 2
      END IF
*
* 162: Largest initial number of vectors in iterative subspace
*
      IF(ISETKW(162).EQ.0) THEN
        MXVC_I = MXCIVG
        ISETKW(162) = 2
      END IF
*
* 163: Largest number of macroiterations
*
      IF(ISETKW(163).EQ.0) THEN
        MAXITM = 3
        ISETKW(163) = 2
      END IF
*
* 164: Fix internal in internal contraction calc: default is no way
*
      IF(ISETKW(164).EQ.0) THEN
        I_FIX_INTERNAL = 0
        ISETKW(164) = 2
      END IF
*
* 165: Form of Hamiltonian for internal zero-order states: Default is 1-body op
*
      IF(ISETKW(165).EQ.0) THEN
        I_INT_HAM = 1
        ISETKW(165) = 2
      END IF
*
*. 166: Use EI-approach for internal: default is yes
*
      IF(ISETKW(166).EQ.0) THEN
        I_DO_EI = 1
        IEI_VERSION = 1
        ISETKW(166) = 2
      END IF
*
*. 167: Excitation in old form. Required for CI SD(TQ) etc.
*
      IF(ISETKW(167).EQ.0.AND.I_DO_EI.EQ.0) THEN
        WRITE(6,*) 
     &  ' Error: exc. info should be supplied in old form for NO_EI'
        NERROR = NERROR + 1
        NMISS = NMISS + 1
        ISETKW(167) = -1 
       END IF
*
*. 168: Information for general(multiop) internal contraction
*       must have been defined if GICCI will be called
*
      IF(I_DO_GIC .EQ. 1.AND.ISETKW(168).EQ.0) THEN
        WRITE(6,*) ' GICCI will be called, requires keyword GIC_EX'
        NERROR = NERROR + 1
        NMISS = NMISS + 1
        ISETKW(168) = -1
      END IF
*
*. 169: Default print level for MCSCF
*
      IF(ISETKW(169).EQ.0) THEN
        IPRMCSCF = 2
        ISETKW(169) = 2
      END IF
* 
*. 170: Method for MCSCF optimization
*
      IF(ISETKW(170).EQ.0) THEN
* IMCSCF_MET = 2 => Two step procedure 
         IMCSCF_MET = 2
*. Orbital E2 is constructed 
         IOOE2_APR = 1
*. Linesearch will be performed
         I_DO_LINSEA_MCSCF = 1
*
         ISETKW(170) = 2
       END IF
*
*. 171: Method for storing and transforming two-eletron integrals
*
       IF(ISETKW(171).EQ.0) THEN
*. Default is New form
          ITRA_ROUTE = 2
         ISETKW(171) = 2
       END IF
*. Well, if old method was specified, but Nonorthogonal CI is 
* in action, switch to new approach
       IF(ISETKW(171).EQ.1.AND.ITRA_ROUTE.EQ.1.AND.
     &    I_DO_NORTCI.EQ.1) THEN
          ITRA_ROUTE = 2
          WRITE(6,*) 
     &    ' New approach for storing integrals will be used'
          WRITE(6,*) 
     &    ' since non-orthogonal CI calculation will be done '
          WRITE(6,*) '                  Best Wishes '
          WRITE(6,*) '                     Lucia    '
       END IF
*
* 172: Info on nonorthogonal CI space: No defaults
*
       IF(ISETKW(172).EQ.0.AND.I_DO_NORTCI.EQ.1) THEN
         WRITE(6,*) ' Input error: Keyword NORTIN is missing '
         NERROR = NERROR + 1
         ISETKW(172) = -1
       END IF
*
* 173: Info in reference CI space for non-orthogonal CI: No defaults
*
       IF(ISETKW(173).EQ.0.AND.I_DO_NORTCI.EQ.1) THEN
         WRITE(6,*) ' Input error: Keyword VBRFSP is missing '
         NERROR = NERROR + 1
         ISETKW(173) = -1
       END IF
*
* 174: ALgorithm for non-orthogonal CI: Default is reexpand in full space
*
      IF(ISETKW(174).EQ.0) THEN
       NORT_MET = 1
       ISETKW(174) = 2
      END IF
*
* 175: Read in information on fragment MO's: Default is no way
*
      IF(ISETKW(175).EQ.0) THEN
       NFRAG_TP = 0
       ISETKW(175) = 2
      END IF
*
* 176: Fragment to Molecule MO's: Should be given in fragment calc
*
      IF(ISETKW(176).EQ.0.AND.ISETKW(175).EQ.1) THEN
        WRITE(6,*) ' Fragment basis specified, but no FRAGOB keyword'
        WRITE(6,*) ' Specify FRAGOB'
        NERROR = NERROR + 1
        ISETKW(176) = -1
       END IF
*
* 177: Choice of Initial set of orbitals
*      Should be defined if Non-orthogonal CI calculation is in action
      IF(ISETKW(177).EQ.0) THEN
        IF(I_DO_NORTCI.EQ.1) THEN
          WRITE(6,*) 
     &    ' Keyword INI_MO should be specified for nonort. calc'
          ISETKW(177) = -1
          NERROR = NERROR + 1
        ELSE
*. Default is read in from environment, no orthogonalization
          INI_MO_TP = 3
          INI_MO_ORT = 0
          INI_ORT_VBGAS = 0
          ISETKW(177) = 2
        END IF
      END IF
*
* 178: Initial Configuration: No default
*
      IF(ISETKW(178).EQ.0) THEN
        I_HAVE_INI_CONF = 0
        ISETKW(178) = - 1
      END IF
*
* 179: Print flag for VB calculation
*
      IF(ISETKW(179).EQ.0) THEN
        IPRVB = 2
        ISETKW(179) = -1
      END IF
*
* 180: Giovannis BK-like CI
*
      IF(ISETKW(180).EQ.0) THEN
        I_AM_GIOVANNI = 0
        ISETKW(180) = 2
      END IF
*
* 181: Print flag for integrals
*
      IF(ISETKW(181).EQ.0) THEN
       IPRINTEGRAL = 0
       ISETKW(181) = 2
      END IF
*
* 182: Preconditioner in CSF basis
*
      IF(ISETKW(182).EQ.0) THEN
*. Default is averaged determinant diagonal
       IH0_CSF = 1
       ISETKW(182) = 2
      END IF
*
* 183: Storage mode for CNF's info and expansions
*      Default is initially storage for all occupation classes
*
      IF(ISETKW(183).EQ.0) THEN
        ICNFBAT = 1
        ISETKW(183) = 2
      END IF
*
* 184: Ensembles of Gaspaces: Default is no
*
      IF(ISETKW(184).EQ.0) THEN
        NENSGS = 0
        ISETKW(184) = 2
      END IF
*
* 185: Constraints on the number of electron in ensemble GAS 1: 
*      default is no (flagged by  I_CHECK_ENSGS = 0)
*
      IF(ISETKW(185).EQ.0) THEN
        I_CHECK_ENSGS = 0
        DO ISPC = 1, NCISPC
          NELVAL_IN_ENSGS(ISPC) = -1
        END DO
        ISETKW(185) = 2
      END IF
*
* 186: Equivalent groups of fragments, default is all fragments are separate groups
*
      IF(ISETKW(186).EQ.0) THEN
        ISETKW(186) = 2
        NEQVGRP_FRAG = NFRAG_MOL
        DO IFRAG = 1, NFRAG_MOL
          LEQVGRP_FRAG(IFRAG) = 1
          IEQVGRP_FRAG(1,IFRAG) = IFRAG
        END DO
      END IF
*
* 187: Supersymmetry, default is no way
*
      IF(ISETKW(187).EQ.0) THEN
        ISETKW(187) = 2 
        I_USE_SUPSYM = 0
        CSUPSYM(1:6) = '      '
      END IF
*
* 188: Specification of Doubly occupied irreps for HF with super symmetry
*
      IF(ISETKW(188).EQ.0) THEN
        ISETKW(188) = 2
        IZERO = 0
        CALL ISETVC(NHFD_IRREP_SUPSYM,IZERO,MAX_SUPSYM_IRREP)
      END IF
*
* 189: Specification of Singly occupied irreps for HF with super symmetry
*
      IF(ISETKW(189).EQ.0) THEN
        ISETKW(189) = 2
        IZERO = 0
        CALL ISETVC(NHFS_IRREP_SUPSYM,IZERO,MAX_SUPSYM_IRREP)
*. If Hartree-Fock calculation will be performed, but neither
*. singly or doubly occupied orbitals have been specified, 
*. we have an error
        IF(ISETKW(188).EQ.2.AND.I_DO_HF.EQ.1) THEN
          NERROR = NERROR + 1
          WRITE(6,*) ' HF calculation specified '
          WRITE(6,*)
     &    ' but neither singly or doubly occupied orbitals specified'
        END IF
      END IF
*
* 190: Specification of irreps of the GASpaces, including 0 and NGAS + 1
*       Must pt be read in if supersymmetry is active
*
      IF(ISETKW(190).EQ.0) THEN
        ISETKW(190) = 2
        IF(I_USE_SUPSYM.EQ.1) THEN
          WRITE(6,*) ' GAS_SP must be specified for supersymmetry'
          NERROR = NERROR + 1
          NMISS = NMISS + 1
          ISETKW(190) = -1
        END IF
        IZERO = 0
        DO IGAS = 0, NGAS + 1
         CALL ISETVC(NGAS_IRREP_SUPSYM(1,IGAS),IZERO,MAX_SUPSYM_IRREP)
        END DO
      END IF
*
* 191: Neglect reordering to standard supersymmetry order in final orbitals: Default is NO
*
      IF(ISETKW(191).EQ.0) THEN
        I_NEGLECT_SUPSYM_FINAL_MO = 0
        ISETKW(191) = 2
      END IF
*
* 192: Use LUCIA fragment files even if overall environment is DALTON, default is no
      IF(ISETKW(192).EQ.0) THEN
        I_USE_LUCIA_FRAGMENTS = 0
        ISETKW(192) = 2
      END IF
*
* 193: Freeze orbitals in MCSCF optimization
*
      IF(ISETKW(193).EQ.0) THEN
        NFRZ_ORB = 0
        ISETKW(193) = 2
      END IF
*
* 194: No CI space where exact Hamiltonian is used in subspace
*
       IF(ISETKW(194).EQ.0) THEN
         ISBSPPR = 0
         ISBSPPR_INI = 0
         ISETKW(194) = 2
       END IF
*
* 195: Number of roots to be converged in first CI: default is NROOT
*     
       IF(ISETKW(195).EQ.0) THEN
         INI_NROOT = NROOT
         ISETKW(195) = 2
       END IF
*
* 196: Roots to be selected from initial CI: Default is root number NROOT
*
       IF(ISETKW(196).EQ.0) THEN
         IROOT_MET(1:6) = 'SELORD'
         ITG_SROOT = NROOT
         ISETKW(196) = 2
*. This is done in all iteration
         
       END IF
*
* 197: Root selection, Default is no
*
       IF(ISETKW(197).EQ.0) THEN
         IROOT_SEL = 0
         ISETKW(197) = 2
       END IF
*
* 198: First space where orbitals are frozen: default is the first space
*
      IF(ISETKW(198).EQ.0) THEN
        IFRZFST = 1
        ISETKW(198) = 2
      END IF
*
* 199: Additional VB orbital spaces, default is not
*
      IF(ISETKW(199).EQ.0) THEN
        NVBGNSP = 0
        ISETKW(199) = 2
      END IF
*
* 200: Order of orbitals for min max spaces, default is same as input
*
      IF(ISETKW(200).EQ.0) THEN
C            ISTVC2(IVEC,IBASE,IFACT,NDIM)
        CALL ISTVC2(IREO_MNMX_OB_NO,0,1,NORBVBSPC)
        CALL ISTVC2(IREO_MNMX_OB_ON,0,1,NORBVBSPC)
        ISETKW(200) = 2
      END IF
*
* 201: Order for which orbitals are coupled in the nort calculations
*
      IF(ISETKW(201).EQ.0) THEN
C            ISTVC2(IVEC,IBASE,IFACT,NDIM)
        CALL ISTVC2(IREO_SPCP_OB_NO,0,1,NORBVBSPC)
        CALL ISTVC2(IREO_SPCP_OB_ON,0,1,NORBVBSPC)
        ISETKW(201) = 2
      END IF
*
* 202: Method for AKBK, standard is PT disc intensive version
*
      IF(ISETKW(202).EQ.0) THEN
         IAKBK_MET = 1
         ISETKW(202) = 2
      END IF

*
*. Largest number of active irreps in super or normal symmetry
      IF(I_USE_SUPSYM.EQ.0) THEN
        NACT_SUPSYM_IRREP = NIRREP
      END IF
*
*. End of reading keywords
* ==========================
*
*. Thresholds only active in connection with IDIAG = 2,
*. Check and maybe issue a warning
      IF(IDIAG.EQ.2) THEN
*. Check to ensure that zero or two thresholds were  set,
        IF(ISETKW(69).NE.ISETKW(70)) THEN
          WRITE(LUOUT,*) 
     &    ' Only a single threshold (E_THRE or C_THRE) '
          WRITE(LUOUT,*)  
     &    ' on individual determinants given. '            
          WRITE(LUOUT,*)  
     &    ' One of the thresholds vanishes therefore and ' 
          WRITE(LUOUT,*)  
     &    ' all determinants will therefore be included  ' 
          WRITE(LUOUT,*)
          WRITE(LUOUT,*) '                   Warns '     
          WRITE(LUOUT,*)
          WRITE(LUOUT,*) '                   LUCIA  '     
        END IF
      ELSE
*. Good old diagonalization, thrsholds not active
        IF(ISETKW(69).EQ.1.OR.ISETKW(70).EQ.1) THEN
          WRITE(LUOUT,*)
     &    ' Thresholds on selection of individual coefficients '
          WRITE(LUOUT,*)
     &    ' are only active in connection with keyword TERACI  '
          WRITE(LUOUT,*)
          WRITE(LUOUT,*) '                   Warns '     
          WRITE(LUOUT,*)
          WRITE(LUOUT,*) '                   LUCIA  '     
        END IF
      END IF
*
      IF(ISETKW(156).EQ.0.AND.ISETKW(50).EQ.1.AND. ISETKW(51).EQ.0) THEN
* Number of GAS shells given but no occupations !!
*. This is okay if we are playing arounf with product wf (therefore the 155 test), else not
        WRITE(6,*) ' GAS calculation (GASSH specified)'
        WRITE(6,*) ' But no Occupation constraints (GASSPC) '
        WRITE(6,*) 
        WRITE(6,*) ' Please add GASSPC '
        NMISS = NMISS + 1
      END IF
*
* 
*
      IF(NMISS.NE.0.OR.NERROR.NE.0 ) THEN
        WRITE(LUOUT,'(1H ,A,I9)')
     &  ' Number of missing required keyword ', NMISS
        WRITE(LUOUT,'(1H ,A,I9)')
     &  ' Number of errors in input ', NERROR
        WRITE(LUOUT,*)
     &  ' You have wounded me I give up '
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
        WRITE(LUOUT,*)
     & '     An expert is a man who has made all the mistakes,'
        WRITE(LUOUT,*)
     &  '     which can be made, in a very narrow field        '
        WRITE(LUOUT,*)
     &  '                                                      '
        WRITE(LUOUT,*)
     &  '                                      Niels Bohr      '
        IF(IEXPERT.EQ.0) THEN
          STOP
        ELSE
          WRITE(6,*) ' Processing continues (EXPERT mode )'
        END IF
      END IF
*. If requested so, we reduce the symmetry at this point
      IF (SYMRED) THEN
        IF (NIRREP.NE.NIRREP_OLD) THEN
          WRITE(6,*) 'LUCIA.SDC does not fit with current run.'
          WRITE(6,*) 'I dicided to ignore it .....'
          SYMRED = .FALSE.
        ELSE
          ! all info is coming from common /symrdc/
          NIRREP = NIRREP_NEW
          NSMOB = NIRREP_NEW
          NSMCMP = NIRREP_NEW
          ! inactive and deleted
          CALL REOSYM(NINASH,NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          CALL REOSYM(NDELSH,NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          ! old RASpaces
          CALL REOSYM(NRS0SH,NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          CALL REOSYM(NRSSH(1,1),NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          CALL REOSYM(NRSSH(1,2),NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          CALL REOSYM(NRSSH(1,3),NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          CALL REOSYM(NRS4SH,NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT) 
          ! new GASpaces
          DO IGAS = 1, NGAS
            CALL REOSYM(NGSSH(1,IGAS),NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT)
          END DO
          IF (IFINMO.EQ.5) THEN
            DO IPSSPC = 1, NPSSPC
              CALL REOSYM(NPSSH(1,IPSSPC),NIRREP_NEW,
     &                                    NIRREP_OLD,IRMAP,NQUOT) 
            END DO
          END IF
          ! for CC excitations
          IF (I_DO_CC_EXC_E.EQ.1)
     &       CALL REOSYM(NEXC_PER_SYM,NIRREP_NEW,NIRREP_OLD,IRMAP,NQUOT)
        END IF
      END IF
 
*. Open one-electron file to obtain core energy and
*. Number of MO's and AO's
      IF(NOINT.EQ.0.AND.
     &   (ENVIRO(1:4).NE.'NONE'.AND.ENVIRO(1:4).NE.'FUSK')
     &   .AND.ENVIRO(1:6).NE.'LIPKIN') THEN
        CALL GET_ORB_DIM_ENV(ECORE_ENV)
        IF(ISETKW(46).EQ.2) ECORE = ECORE_ENV
        CALL CHK_ORBDIM(IGSFILL,ISECFILL)
      ELSE
        WRITE(6,*) ' GETOBS and CHK_ORBDIM not called '
        ECORE = 0.0D0
      END IF
*. If a minmax subspace has been defined, extend occupations to
*. all active orbitals
      IF(ISBSPC_SEL.EQ.4) THEN
        MINEL_L = ISBSPC_MINMAX(NSBSPC_ORB,1)
        MAXEL_L = ISBSPC_MINMAX(NSBSPC_ORB,2)
        DO IORB = NSBSPC_ORB+1, NACOB-1
          ISBSPC_MINMAX(IORB,1) = MINEL_L
          ISBSPC_MINMAX(IORB,2) = MAXEL_L
        END DO
        ISBSPC_MINMAX(NACOB,1) = NACTEL
        ISBSPC_MINMAX(NACOB,2) = NACTEL
*.  And the orbitals: Orbitals following NSBSPC_ORB are not changed
        DO IOB = 1, NSBSPC_ORB
         ISBSPC_ORB_INV(ISBSPC_ORB(IOB)) = IOB
        END DO
        DO IOB = NSBSPC_ORB + 1, NACOB
          ISBSPC_ORB(IOB) = IOB
          ISBSPC_ORB_INV(IOB) = IOB
        END DO
      END IF

*. Check to see if there a CI calculation will be called after 
*. CC, as this will enforce CC=>CI expansion
      I_DO_CI_AFTER_CC = 0
      I_HAVE_DONE_CC = 0
             CARDX=ITEM(1)
             CSEQCI(ICI,JCMBSPC) = ITEM(1)(1:8)
*
      DO JCMBSPC = 1, NCMBSPC
        DO ICI = 1, NSEQCI(JCMBSPC)
           CARDX=CSEQCI(ICI,JCMBSPC)
           
           IF(CARDX(1:2).EQ.'CC'.OR.CARDX(1:6).EQ.'GEN_CC') THEN
              I_HAVE_DONE_CC = 1
           END IF
           IF(CARDX(1:2).EQ.'CI'.AND.I_HAVE_DONE_CC.EQ.1) THEN
             I_DO_CI_AFTER_CC = 1
           END IF
        END DO
      END DO
C?    WRITE(6,*) ' Check,  I_DO_CI_AFTER_CC = ',  I_DO_CI_AFTER_CC
C?    WRITE(6,*) ' Check,  I_DO_NEWCCV = ',  I_DO_NEWCCV
      IF( I_DO_CI_AFTER_CC .EQ. 1 ) THEN
*. Enforce CC => CI expansion
        I_DO_CC_TO_CI = 1
      END IF
*
*. Initial order of MOs is presently not known
      CMO_ORD = 'UNK'
  
   
*. Check number of orbitals and insert occupations for ALL/REST
 
************************************************************
*                                                          *
* Part 3: Print input                                     *
*                                                          *
************************************************************
*
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '******************'
      WRITE(LUOUT,*) '*  Title of run  *'
      WRITE(LUOUT,*) '******************'
      WRITE(LUOUT,*)
      CALL PRTITL(TITLEC)
      WRITE(LUOUT,*)
*
*. Machine in use
      WRITE(6,'(A,A)') '    Machine in use: ', MACHINE
*
*. Core memory
      WRITE(6,'(A,I20,A)')
     &                 '    Core memory   : ', MAXMEM,' R*8 words'
*
*. Type of reference state
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '********************************'
      WRITE(LUOUT,*) '*  Symmetry and spin of states *'
      WRITE(LUOUT,*) '********************************'
      WRITE(LUOUT,*)
*. Point group
      IF(PNTGRP.EQ.1) THEN
        IF (NIRREP.EQ.8) THEN
          WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ D2h'
        ELSE IF (NIRREP.EQ.4) THEN
          WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ C2v/C2h/D2'
        ELSE IF (NIRREP.EQ.2) THEN
          WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ Cs/Ci/C2'
        ELSE IF (NIRREP.EQ.1) THEN
          WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ C1'
        END IF
      ELSE IF(PNTGRP.EQ.2) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ C inf v'
      ELSE IF(PNTGRP.EQ.3) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ D inf h'
      ELSE IF(PNTGRP.EQ.4) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Point group ............ O3'
      END IF
*
      IF(I_USE_SUPSYM.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,A)')
     &  '     Super symmetry ......... ', CSUPSYM
      END IF
*.Spatial symmetry
      IF(PNTGRP.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,I1)')
     &  '     Spatial symmetry ....... ', IREFSM
      ELSE IF(PNTGRP.EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I1)')
     &  '     ML value ............... ', IREFML
      ELSE IF(PNTGRP.EQ.3) THEN
        WRITE(LUOUT,'(1H ,A,I1)')
     &  '     ML value ............... ', IREFML
        IF(IREFPA.EQ.1) WRITE(LUOUT,'(1H ,A)')
     &  '     Parity   ..............  Gerade'
        IF(IREFPA.EQ.2) WRITE(LUOUT,'(1H ,A)')
     &  '     Parity   ..............  Ungerade'
      ELSE IF(PNTGRP.EQ.4) THEN
        WRITE(LUOUT,'(1H ,A,I1)')
     &  '     L  value ............... ', IREFL
        WRITE(LUOUT,'(1H ,A,I1)')
     &  '     ML value ............... ', IREFML
        IF(IREFPA.EQ.1) WRITE(LUOUT,'(1H ,A)')
     &  '     Parity   ..............  Gerade'
        IF(IREFPA.EQ.2) WRITE(LUOUT,'(1H ,A)')
     &  '     Parity   ..............  Ungerade'
      END IF
*.Spin
      WRITE(LUOUT,'(1H ,A,I2)')
     &  '     2 times spinprojection  ', MS2
*. Intermediate Spin projection 
      IF(I_RE_MS2_SPACE.NE.0) THEN
        WRITE(LUOUT,'(1H ,A,I2,A,I2)')
     &  '     2*MS2 after orbital space ', I_RE_MS2_SPACE, 
     &  ' must be ',I_RE_MS2_VALUE
      END IF
      IF(NOCSF.EQ.0) WRITE(LUOUT,'(1H ,A,I2)')
     &  '     Spin multiplicity ....  ', MULTS
*.Number of active electrons
      WRITE(LUOUT,'(1H ,A,I2)')
     &  '     Active electrons .....  ', NACTEL
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '*********************************************'
      WRITE(LUOUT,*) '*  Shell spaces and occupation constraints  *'
      WRITE(LUOUT,*) '********************************************* '
      WRITE(LUOUT,*)
*
      IF(IDOGAS.EQ.0) THEN
*. Kept because output can lated be used for GAS
*
*. NOT a GAS expansion
*
*
      WRITE(LUOUT,'(1H ,A,10I4)')
     &  '                Irrep ',(I,I = 1,NIRREP)
      WRITE(LUOUT,'(1H ,A,2X,10A)')
     &  '                ===== ',('====',I = 1,NIRREP)
*
*. Inactive
      IF(ISETKW(7).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Inactive         ',(NINASH(I),I=1,NIRREP)
      END IF
*. Core
      IF(ISETKW(8).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Core             ',(NRS0SH(I,1),I=1,NIRREP)
      END IF
*. RAS1
      IF(ISETKW(9).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Ras1             ',(NRSSH(I,1),I=1,NIRREP)
      END IF
*. RAS2/ACTIVE
      IF(ISETKW(10).EQ.1) THEN
        IF(INTSPC.EQ.1) THEN
          WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Active           ',(NRSSH(I,2),I=1,NIRREP)
        ELSE IF(INTSPC.EQ.2) THEN
          WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Ras2             ',(NRSSH(I,2),I=1,NIRREP)
        END IF
      END IF
*. RAS3
      IF(ISETKW(11).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Ras3             ',(NRSSH(I,3),I=1,NIRREP)
      END IF
      IF(INTSPC.EQ.2.AND.IMLCR3.EQ.1) WRITE(LUOUT,'(1H ,A)')
     &  '     ( RAS 3 space supplied by courtesy of TRAONE )'
*. Secondary space
      IF(ISETKW(13).EQ.1) THEN
        DO 310 ITP = 1,MXR4TP
          WRITE(LUOUT,'(1H ,A,I2,A,10I4)')
     &  '     Secondary',ITP,'      ',(NRS4SH(I,ITP),I=1,NIRREP)
  310   CONTINUE
      END IF
*. Deleted space
      IF(ISETKW(26).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,10I4)')
     &  '     Deleted          ',(NDELSH(I),I=1,NIRREP)
      END IF
      IF(IMLCR3.EQ.2) WRITE(LUOUT,'(1H ,A)')
     &  '     ( Deleted shells supplied by courtesy of TRAONE )'
*.Core space
      WRITE(LUOUT,*)
      IF(ISETKW(8).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,I2)')
     &  '     Largest number of excitations out of core .....   ',MXHR0
      END IF
*.Secondary space
      IF(ISETKW(13).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,I2)')
     &  '     Largest number of excitations to secondary space  ',MXER4
      END IF
      ELSE
*
*. GAS space
*
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) ' *************************'
      WRITE(LUOUT,*) ' Generalized active space '
      WRITE(LUOUT,*) ' *************************'
      WRITE(LUOUT,*)
      WRITE(LUOUT,'(A)') ' Orbital subspaces:'
      WRITE(LUOUT,'(A)') ' ================== '
      WRITE(LUOUT,*)
      WRITE(LUOUT,'(1H ,A,10I4,A)')
     &  '                Irrep ',(I,I = 1,NIRREP) 
      WRITE(LUOUT,'(1H ,A,2X,10A,A)')
     &  '                ===== ',('====',I = 1,NIRREP) 
      WRITE(LUOUT,'(A,10I4)')
     &  '        Inactive       ',
     &   ( NINASH(IRREP),IRREP = 1, NIRREP)
   
      DO IGAS = 1, NGAS
        WRITE(LUOUT,'(A,I2,A,10I4,6X,2I6)')
     &  '        GAS',IGAS,'          ', 
     &  (NGSSH(IRREP,IGAS),IRREP = 1, NIRREP) 
      END DO
      WRITE(LUOUT,'(A,10I4)')
     &  '        Secondary      ',
     &   ( NSECSH(IRREP),IRREP = 1, NIRREP)
      WRITE(6,*)
      IF(IGSFILL.NE.0) WRITE(6,'(7X,A,I3)')
     &' Gas space provided by courtesy of LUCIA:',  IGSFILL 
      IF(ISECFILL.NE.0) WRITE(6,'(7X,A,I3)')
     &' Secondary space provided by courtesy of LUCIA '
*
      IF(I_USE_SUPSYM.EQ.1) THEN
        WRITE(6,*)
        WRITE(6,*) ' Information on supersymmetry irreps '
        WRITE(6,*) ' ************************************'
        WRITE(6,*)
        IF(ISETKW(188).EQ.1) THEN
          WRITE(6,'(A,15(1X,I2))') 
     &   ' Doubly occupied irreps in HF: ',
     &     (NHFD_IRREP_SUPSYM(I), I = 1, NACT_SUPSYM_IRREP)
        END IF
        IF(ISETKW(189).EQ.1) THEN
          WRITE(6,'(A,15(1X,I2))') 
     &   ' Singly occupied irreps in HF: ',
     &     (NHFS_IRREP_SUPSYM(I), I = 1, NACT_SUPSYM_IRREP)
        END IF
        IF(ISETKW(190).EQ.1) THEN
         WRITE(6,*)
     &   ' Irreps in the various GASpaces: '
         WRITE(6,*) ' Gas, Number of sets per irrep '
         WRITE(6,*) ' =================================='
         DO IGAS = 0, NGAS + 1
           WRITE(6,'(1X,I3,4X,20(1X,I3))')
     &     IGAS, (NGAS_IRREP_SUPSYM(I,IGAS), I = 1, NACT_SUPSYM_IRREP)
         END DO
       END IF
      END IF
         
*
*
      IF(IDOQD.EQ.1)   WRITE(6,'(7X,A,I3)')
     &' Number of orbital spaces used for holes ', N_HOLE_ORBSPACE 
*
      IF(I_DO_PRODEXP.EQ.0) THEN
       WRITE(LUOUT,*)
       WRITE(LUOUT,*)  '*******************'
       WRITE(LUOUT,*)  ' Occupation spaces '
       WRITE(LUOUT,*)  '*******************'
       WRITE(LUOUT,*)
       WRITE(LUOUT,'(A,I3)')
     &  ' Number of Occupation spaces: ',NCISPC
       WRITE(LUOUT,*)
       DO ICISPC = 1, NCISPC
        WRITE(LUOUT,'(A,I3)') 
     &  ' Bounds on accumulated occupations for space: ',ICISPC
        WRITE(LUOUT,'(A)')
     &   ' ====================================================== '
        WRITE(LUOUT,'(A)')
        WRITE(LUOUT,'(A)') '         Min. occ    Max. occ '
        WRITE(LUOUT,'(A)') '         ========    ======== '
        DO IGAS = 1, NGAS
          WRITE(LUOUT,'(A,I2,3X,I3,9X,I3)')
     &    '   GAS',IGAS,IGSOCCX(IGAS,1,ICISPC),IGSOCCX(IGAS,2,ICISPC)
        END DO
       END DO
*
       IF(ISETKW(14).EQ.1) THEN
         WRITE(LUOUT,'(A,I3)') 
     &   ' Bounds on accumulated occupations for reference space '
         WRITE(LUOUT,'(A)')
     &    ' ====================================================== '
         WRITE(LUOUT,'(A)')
         WRITE(LUOUT,'(A)') '         Min. occ    Max. occ '
         WRITE(LUOUT,'(A)') '         ========    ======== '
         DO IGAS = 1, NGAS
           WRITE(LUOUT,'(A,I2,3X,I3,9X,I3)')
     &     '   GAS',IGAS,IREFOCC_ACC(IGAS,1),IREFOCC_ACC(IGAS,2)
         END DO
       END IF
       
      IF(NENSGS.NE.0) THEN
       WRITE(6,'(7X,A)')
     & 'Ensembles of GASpaces: '
       WRITE(6,'(7X,A)')
     & '====================== '
       DO JENSGS = 1, NENSGS
         WRITE(6,'(10X,A,I3)')
     & ' Ensemble: ', JENSGS
         N = LENSGS(JENSGS)
         WRITE(6,'(10X,30(1X,I2))')
     &   (IENSGS(IGAS,JENSGS),IGAS = 1, N)
       END DO
      END IF
*
      IF(I_CHECK_ENSGS.EQ.1) THEN
        DO ISPC = 1, NCISPC
          WRITE(6,'(10X,A,I3)') 
     &    ' Allowed numbers of electrons in Ensemble GAS I for space:',
     &    ISPC 
          WRITE(6,'(10X,20(1X,I3))')
     &    (IEL_IN_ENSGS(IVAL,ISPC),IVAL = 1, NELVAL_IN_ENSGS(ISPC))
        END DO
      END IF
*
       IF(ISETKW(52).EQ.1) THEN
        WRITE(LUOUT,*)
        WRITE(LUOUT,*) 
     &  ' **************************************************'
        WRITE(LUOUT,*) 
     &  ' Specification of CI Spaces (combinations of above)'
        WRITE(LUOUT,*) 
     &  ' **************************************************'
        WRITE(LUOUT,*)
   
        WRITE(6,*) 
        WRITE(6,'(A,I3)')
     &  ' Number of CI spaces included: ', NCMBSPC
        WRITE(6,*) 
        DO JCMBSPC = 1, NCMBSPC
          WRITE(6,*)
          WRITE(6,'(A,I3)') ' Information about CI space ', JCMBSPC
          WRITE(6,'(A)')    ' =================================='
          WRITE(6,'(1H ,3X,A,I3)')
     &    'Number of occupation spaces included  ',LCMBSPC(JCMBSPC)    
          WRITE(6,'(A,10I3)') '    Occupation spaces included ', 
     &    (ICMBSPC(II,JCMBSPC),II=1,LCMBSPC(JCMBSPC))
*
        END DO 
       END IF
      ELSE
*. Product type of expansion
      END IF

*     

      WRITE(LUOUT,*)
      WRITE(LUOUT,*) ' ******************************************'
      WRITE(LUOUT,*) ' Specification of Sequence of calculations '
      WRITE(LUOUT,*) ' ******************************************'
      WRITE(LUOUT,*)
      DO JCMBSPC = 1, NCMBSPC
        WRITE(6,*)
        WRITE(6,'(7X,A,I3)') ' Space ', JCMBSPC
        WRITE(6,'(7X,A)')    ' =============='
        WRITE(6,*)
*
C       WRITE(6,'(A,I3)') ' Number of calculations in this space ',
C    &  NSEQCI(JCMBSPC)
C       WRITE(6,'(A)')   '  Calculations in this space '
C       WRITE(6,'(A)')   '  ==========================='
        DO JSEQ = 1, NSEQCI(JCMBSPC)
          CARDX = CSEQCI(JSEQ,JCMBSPC)
          IF(CARDX(1:7).EQ.'VECFREE') THEN
            WRITE(6,'(10X,A,I3)') 
     &      '       Vector free calculation at level ',
     &      -ISEQCI(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:2).EQ.'CI') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       Normal CI with max. iterations = ', 
     &      ISEQCI(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:6).EQ.'APR-CI') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       CI using approximate H with max. iterations = ', 
     &      ISEQCI(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:5).EQ.'PERTU') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       Perturbation calculation '          
          ELSE IF(CARDX(1:2).EQ.'CC'   ) THEN
            WRITE(6,'(10X,A,I3)')
     &      '       Coupled Cluster Calculation, max. iterations =',      
     &      ISEQCI(JSEQ,JCMBSPC)
            IF(JCMBSPC.EQ.LAST_CC_SPC.AND.JSEQ.EQ.LAST_CC_RUN) THEN
              WRITE(6,'(10X,A)') 
     &      '       (Expanded cc wf will be transferred to LUC ) '
            END IF
          ELSE IF(CARDX(1:6).EQ.'GEN_CC'.OR.
     &            CARDX(1:3).EQ.'TCC') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       General Coupled Cluster, max. iterations =',      
     &      ISEQCI(JSEQ,JCMBSPC)
            WRITE(6,'(10X,A,I3)')
     &      '                                Operator space  =',      
     &      ISEQCI2(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:3).EQ.'UCC') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       Unitary Coupled Cluster, max. iterations =',      
     &      ISEQCI(JSEQ,JCMBSPC)
            WRITE(6,'(10X,A,I3)')
     &      '                                Operator space  =',      
     &      ISEQCI2(JSEQ,JCMBSPC)
c            STOP 'program adaption not complete'
          ELSE IF(CARDX(1:3).EQ.'VCC') THEN
            WRITE(6,'(10X,A,I3)')
     &      '       Variational Coupled Cluster, max. iterations =',      
     &      ISEQCI(JSEQ,JCMBSPC)
            WRITE(6,'(10X,A,I3)')
     &      '                                Operator space  =',      
     &      ISEQCI2(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:4).EQ.'ICCI' ) THEN
            WRITE(6,'(10X,A)') 
     &      '       Internal Contracted CI calculation '  
          ELSE IF(CARDX(1:4).EQ.'ICCC' ) THEN
            WRITE(6,'(10X,A)') 
     &      '       Internal Contracted CC calculation (aka MRCC) '  
            IF(IREADSJ.EQ.1) WRITE(6,'(10X,A)') 
     &      '       Metric and approximate Jacobian will be read in '
          ELSE IF(CARDX(1:7).EQ.'TWOBODY' ) THEN
            WRITE(6,'(10X,A,/,2(10X,A,I4,/))') 
     &      '       Generalized Two-Body operater Cluster expansion',
     &      '         refspace   = ',ISEQCI2(JSEQ,JCMBSPC),
     &      '         max. iter. = ',ISEQCI (JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:6).EQ.'SP_MCL' ) THEN
            WRITE(6,'(10X,A)') 
     &      '       Spin-restricted MCLR '                 
          ELSE IF(CARDX(1:5).EQ.'MCSCF' ) THEN
            WRITE(6,'(10X,A)') 
     &      '       MCSCF optimization  '                 
          ELSE IF(CARDX(1:6).EQ.'NORTCI' ) THEN
            WRITE(6,'(10X,A,I3)') 
     &      '       Nonorthogonal CI with max. iterations = ',
     &      ISEQCI(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:6).EQ.'NORTMC' ) THEN
            WRITE(6,'(10X,A,2I3)') 
     &      '       Nonorthogonal MCSCF,  max. macro and micro its. = ',
     &      ISEQCI(JSEQ,JCMBSPC), ISEQCI2(JSEQ,JCMBSPC)
          ELSE IF(CARDX(1:2).EQ.'HF') THEN
            WRITE(6,'(10X,A,I3)') 
     &      '       Hartree-Fock optimization. iterations = ',
     &      ISEQCI(JSEQ,JCMBSPC)
            WRITE(6,'(10X,A)') 
     &      '       Integrals will be transformed to converged orbitals'
          ELSE IF(CARDX(1:5).EQ.'GICCI') THEN
            WRITE(6,'(10X,A,I4,2x,I4)') 
     &      '       GICCI calculation, outer and inner iterations =',
     &              MAXITM, MAXIT
          ELSE IF(CARDX(1:4).EQ.'AKBK') THEN
            WRITE(6,'(10X,A)') 
     &      '       AKBK aka Split GAS calculation                 ' 
          END IF
        END DO
*       ^ End of loop over spaces in given CI space
      END DO
*     ^ End of loop over CI spaces
      WRITE(6,*) 
      END IF
*     ^ End of GAS/NOGAS switch
*
      IF(XLAMBDA.NE.1.0D0) THEN
        WRITE(6,*)
        WRITE(6,'(A,F13.8)') 
     &  ' Modified operator H(l) = l*F + l*(H-F) used with l =',XLAMBDA
        IF(IUSEH0P.EQ.0) THEN
         WRITE(6,'(A)')  ' Zero-order operator without projection used '
        ELSE
         WRITE(6,'(A)')  ' Zero-order operator with projection used '
        END IF
        IF(IRESTR.EQ.0) THEN
        WRITE(6,*)
     &  ' Notice: This madness starts  in second calculation'
        ELSE
         WRITE(6,*) ' You have specified a calculation with modified '
         WRITE(6,*) ' Hamiltonian (the LAMBDA option) and RESTART '
         WRITE(6,*) ' so this is what I will do '
         WRITE(6,*)   
         WRITE(6,*) '   1:) Perform CI in space 1 to obtain Hamiltonian'
         WRITE(6,*) '       (no RESTART in this space )'
         WRITE(6,*) '   2:) CI calculation in space 2  with '
         WRITE(6,*) '       modified Hamiltonian and RESTART from LU21'
         WRITE(6,*) ' Space 2 should therefore correspond to the'
         WRITE(6,*) ' restarted calculation '
       END IF
      END IF
*
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '***********'
      WRITE(LUOUT,*) '*  Roots  *'
      WRITE(LUOUT,*) '*********** '
      WRITE(LUOUT,*)
      WRITE(LUOUT,'(1H ,A,I3)')
     &  '     Number of roots to be included  ', NROOT
      WRITE(LUOUT,'(1H ,A,(20I3))')
     &  '     Roots to be obtained ', (IROOT(I),I=1, NROOT )
      WRITE(LUOUT,'(1H ,A,I3)')
     &  '     Number of roots to be converged ', NCNV_RT
      IF(INI_NROOT.NE.NROOT) WRITE(LUOUT,'(1H ,A,I3)')
     &  '     Number of roots In initial CI ', INI_NROOT
*
      IF(IROOT_MET(1:6).EQ.'SELORD') THEN
         WRITE(LUOUT,'(1H ,A,I3)')
     &  '     Reference state selected as root ', ITG_SROOT
      ELSE IF(IROOT_MET(1:6).EQ.'SELSPS') THEN
         WRITE(LUOUT,'(1H ,A,I3)')
     &  '     Initial reference state must have supersymmetry ',
     &        ITG_SUPSYM
      END IF
      IF(ISEL_ONLY_INI.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Root selection only in initial calculation '
      ELSE
        WRITE(LUOUT,'(1H ,A)')
     &  '     Root selection in ALL  calculations '
      END IF
*. The above should be rethought...
      IF(IROOT_SEL.EQ.1) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    ' In optimization roots will be selected using root homing '
      ELSE IF (IROOT_SEL.EQ.2) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    ' In optimization roots will be selected using super-symmetry'
      END IF
*
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '**************************'
      WRITE(LUOUT,*) '*  Run time definitions  *'
      WRITE(LUOUT,*) '************************** '
      WRITE(LUOUT,*)
*. Program environment 
      WRITE(6,'(A,A6)')  '      Program environment... ', ENVIRO
*
      IF(IDOQD.EQ.1) THEN
        WRITE(6,'(A,A6)')'      Quantum dot calculation'
      END IF
*. Integral import
      IF(NOINT.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     No integrals will be read in       '
      ELSE IF(NOINT.EQ.0) THEN
*. Quantum dot calculation ?
      IF(IDOQD.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Integrals read in QDOT format '
      ELSE
        IF(INTIMP.EQ.1) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Integrals read in in MOLCAS format '
        ELSE IF(INTIMP.EQ.5) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Integrals read in in SIRIUS format '
        ELSE IF(INTIMP.EQ.2) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Integrals read in in LUCAS format '
        ELSE IF(INTIMP.EQ.3) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Integrals read in in formatted form (E22.15) ',
     &    '      From unit 13'
          WRITE(LUOUT,'(1H ,A)')
     &    '     All integrals of correct symmetry combination read in'
        ELSE IF(INTIMP.EQ.8) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Integrals supplied for Lipkin-quasispin-model'
          WRITE(LUOUT,'(1H ,A,E20.10,/,1H ,A,E20.10)')
     &    '         Parameters: e = ',XLIP_E,
     &    '                     V = ',XLIP_V
        ELSE IF(INTIMP.EQ.9) THEN
          WRITE(LUOUT,'(1H ,A)')
     &    '     Fusk integrals .... '
       END IF
      END IF
*. Integral storage
      IF(INCORE.EQ.1) WRITE(LUOUT,'(1H ,A)')
     &  '     All integrals stored in core'
      END IF
      WRITE(LUOUT,*)
* ( END IF for NOINT 
*. CSF or SD expansion
      IF(NOCSF.EQ.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     CI optimization performed with CSF''s '
      ELSE
        WRITE(LUOUT,'(1H ,A)')
     &  '     CI optimization performed with SD''s '
      END IF
*. Ms,Ml combinations
      IF(ISETKW(27).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,F8.3)')
     &  '     Spin combinations used with sign ',PSSIGN
      END IF
      IF(ISETKW(28).EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,F8.3)')
     &  '     ML   combinations used with sign ',PLSIGN
      END IF
*. Preconditioner for H in CSF basis
      IF(NOCSF.EQ.0) THEN
        IF(IH0_CSF.EQ.1) THEN
          WRITE(LUOUT,'(1H ,4X,A)')
     &    ' CI-diagonal: Averaged Determinant diagonal'
        ELSE IF(IH0_CSF.EQ.2) THEN
          WRITE(LUOUT,'(1H ,4X,A)')
     &    ' CI-diagonal: Diagonal of H in CSF basis'
        ELSE IF(IH0_CSF.EQ.3) THEN
          WRITE(LUOUT,'(1H ,4X,A)')
     &    ' CI-diagonal: Diagonal Configuration blocks of H '
        END IF! Switch over IH0_FORM
      END IF! NOCSF = 0
        
*. Initial approximation to vectors
      WRITE(LUOUT,*)
      IF(IRESTR.EQ.1.AND.IRESTRF.EQ.0) THEN
         WRITE(LUOUT,'(1H ,A)')
     &  '     Restarted calculation '
      ELSE IF(IRESTRF.EQ.1) THEN
         WRITE(LUOUT,'(1H ,A)')
     &  '     Restarted calculation from REFERENCE space expansion'
      ELSE
         IF(ISBSPC_SEL.NE.0) THEN
           WRITE(LUOUT,'(1H ,A)')
     &  '     Initial vectors obtained from explicit Hamiltonian'
         ELSE IF(ISBSPC_SEL.EQ.0) THEN
           WRITE(LUOUT,'(1H ,A)')
     &  '     Initial vectors obtained from diagonal'
         END IF
      END IF
      IF(I_RESTRT_CC.EQ.1) THEN
           WRITE(LUOUT,'(1H ,A)') '     CC calculation restarted '
      END IF
*. Handling of degenerencies of initial vectors
      IF(INIDEG.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Symmetric combination of degenerate initial vectors'
      ELSE IF (INIDEG.EQ.-1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Antiymmetric combination of degenerate initial vectors'
      ELSE IF (INIDEG.EQ.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     No combination of degenerate initial vectors'
      END IF
*. Ms,Ml combinations
C     IF(ISETKW(27).EQ.1) THEN
C       WRITE(LUOUT,'(1H ,A,F8.3)')
C    &  '     Spin combinations used with sign ',PSSIGN
C     END IF
C     IF(ISETKW(28).EQ.1) THEN
C       WRITE(LUOUT,'(1H ,A,F8.3)')
C    &  '     ML   combinations used with sign ',PLSIGN
C     END IF
*. CI storage mode
      WRITE(6,*)
      IF(ICISTR.EQ.1) THEN
        WRITE(6,*)
     &  '     3 symmetry blocks and two vectors will be held in core '
      ELSE IF( ICISTR.EQ.2) THEN
        WRITE(6,*)
     &  '     3 type-type blocks will be held in core '
      ELSE IF( ICISTR.EQ.3) THEN
        IF(ISIMSYM.EQ.0) THEN
        WRITE(6,*)
     &  '     3 type-type-symmetry blocks in core '
        ELSE
        WRITE(6,*)
     &  '     3 type-type-symmetry blocks, all symmetries,  in core '
        END IF
      END IF
*
      IF(NOCSF.EQ.0) THEN
        IF(ICNFBAT.EQ.1) THEN
         WRITE(6,*)
     &  '     All Confs and CSFs treated as a single batch'
        ELSE IF (ICNFBAT.EQ.2)  THEN
         WRITE(6,*)
     &  '     All Confs and CSFs belonging to one occls in a batch'
        END IF
      END IF
*
      IF(LCSBLK.NE.0) WRITE(6,'(A,I10)') 
     &  '      Smallest allowed size of sigma- and C-batch ',LCSBLK
      WRITE(LUOUT,'(1H ,A,I4)')
     &  '     Dimension of block of resolution strings ', MXINKA
      IF(IUSE_PH.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Particle-hole separation used '
      ELSE
        WRITE(LUOUT,'(1H ,A)')
     &  '     Particle-hole separation not used '
      END IF
*
      IF(IADVICE.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Advice routine call to optimize sigma generation'
      END IF
*
      IF(IUSE_PA.EQ.1.OR.ISIMSYM.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Strings divided into active and passive parts'
      ELSE
        WRITE(LUOUT,'(1H ,A)')
     &  '     Strings not divided into active and passive parts'
      END IF
      IF(ISIMSYM.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     ALl TTS blocks with given types treated in sigma'
      END IF
      IF(IUSE_HW .EQ. 1) THEN
        WRITE(6,*) ' Hardwired routines in use '
      END IF
*
      WRITE(LUOUT,*)
      IF(IDENSI.EQ.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     No calculation of density matrices  '            
      ELSE IF(IDENSI.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     One-body density matrix calculated'           
      ELSE IF(IDENSI.EQ.2) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     One- and two-body density matrices  calculated'           
      END IF
      IF(ISPNDEN.EQ.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     No calculation of spin-density matrices  '            
      ELSE IF(ISPNDEN.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     One-body spin-density matrix calculated'           
      ELSE IF(ISPNDEN.EQ.2) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     One- and two-body spin-density matrices  calculated' 
      END IF
*
      IF(IDENSI.GE.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '     Densities will be calculated for active orbitals'
      END IF
*
      IF(ICUMULA.NE.0) THEN
        WRITE(LUOUT,'(1H ,A,I2)')
     &  ' Cumulants will be generated through order', ICUMULA
      END IF
*
      WRITE(LUOUT,*)
C?    IF(MOCAA.NE.0) WRITE(LUOUT,'(1H ,A,I4)')
C?   &  '     MOC method used for alpha-alpha+beta-beta loop '    
C?    IF(MOCAB.NE.0) WRITE(LUOUT,'(1H ,A,I4)')
C?   &  '     MOC method used for alpha-beta loop            '    
*
*. Diagonalization information
      WRITE(LUOUT,'(1H ,A)')
     &  '     CI diagonalization: '
      WRITE(LUOUT,'(1H ,A)')
     &  '     ==================== '
*
*. Subspace Hamiltinian
*
      IF(ISBSPC_SEL.EQ.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &  '        No subspace Hamiltonian '
      ELSE IF(ISBSPC_SEL.EQ.1) THEN
        WRITE(LUOUT, '(1H ,A, I4,A)')
     &  '        Subspace choosen as ', MXP1, 
     &  ' variables with lowest energy'
      ELSE IF(ISBSPC_SEL.EQ.2) THEN
        WRITE(LUOUT, '(1H, A, I4,A)')
     &  ' Subspace choosen as ', MXP1, ' first variables '
      ELSE IF(ISBPSC_SEL.EQ.3) THEN
        WRITE(LUOUT, '(1H, A, I4)')
     &  ' Subspace chosen as CI-space ', ISBSPC_SPC
      ELSE IF(ISBSPC_SEL.EQ.4) THEN
        WRITE(LUOUT, '(1H ,8X, A)') ' Subspace chosen as MINMAX space:'
        CALL WRT_MINMAX_OCC(ISBSPC_MINMAX(1,1), ISBSPC_MINMAX(1,2),
     &       NSBSPC_ORB)
COLD    WRITE(LUOUT,'(1H ,A,3I4)')
COLD &  '        Dimensions of subspace Hamiltonian ',MXP1,MXP2,MXQ
      END IF
*. Diagonalizer
      IF(IDIAG.EQ.1.AND.ICISTR.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
     &    '        Diagonalizer: MINDV4 '
      ELSE IF(IDIAG.EQ.1.AND.ICISTR.GE.2) THEN
        WRITE(LUOUT,'(1H ,A)')
     &    '        Diagonalizer: MICDV* '
      ELSE IF(IDIAG.EQ.2) THEN
      WRITE(LUOUT,'(1H ,A)')
     &    '        Diagonalizer: PICO*  '
      END IF
COLD  IF(NOCSF.EQ.1) THEN
COLD  IF(IPRECOND.EQ.1) THEN
COLD    WRITE(LUOUT,'(1H ,A)')
COLD &      '        Simple diagonal used as preconditioner  '  
COLD  ELSE IF(IPRECOND.EQ.2) THEN
COLD    WRITE(LUOUT,'(1H ,A)')
COLD &  '        Diagonal configuration blocks used as preconditioner'
COLD  END IF
COLD  END IF
*
      IF(ISBSPPR.NE.0) THEN
        WRITE(6,'(A,I2)') 
     &  ' Preconditioner uses exact Hamiltonian in subspace ', ISBSPPR
        WRITE(6,'(A,I2)')
     &  ' Starting from calculations in space ', ISBSPPR_INI
        WRITE(6,'(A)') ' ( Works only for MICDV6 ) '
      END IF
*
*. Root homing
      IF(IROOTHOMING.EQ.1) THEN
      WRITE(LUOUT,'(1H ,A)')
     &  '        Root homing will be used '                           
      ELSE
      WRITE(LUOUT,'(1H ,A)')
     &  '        No root homing '                                   
      END IF
*. No restart in CI calc 2
      IF(IRST2.EQ.0) THEN
      WRITE(LUOUT,'(1H ,A)')
     &  '        No restart from previous vectors in second calc '
      END IF
      IF(ISKIPEI.EQ.1) THEN
      WRITE(LUOUT,'(1H ,A)')
     &  '        Initial energy evaluations skipped after first calc'
      WRITE(LUOUT,'(1H ,A)')
     &  '        (Only active in connection with TERACI )'           
      END IF
*. Number of iterations
C     WRITE(LUOUT,'(1H ,A,I2)')
C    &  '        Allowed number of iterations    ',MAXIT
*. Number of CI vectors in subspace
      WRITE(LUOUT,'(1H ,A,I2)')
     &  '        Allowed Dimension of CI subspace ',MXCIV
* 
      WRITE(LUOUT,'(1H ,A,E11.5)')
     &  '        Convergence threshold for energy ',THRES_E
*. Multispace (multigrid info )
      IF(MULSPC.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A,I3)') 
     &    '        Multispace method in use from space ',
     &             IFMULSPC
        WRITE(6,*) 
     &    '        Pattern '
        CALL IWRTMA(IPAT,1,LPAT,1,LPAT)
      ELSE
        WRITE(LUOUT,'(1H ,A)') 
     &    '        No multispace method in use '
      END IF
*
      WRITE(6,*)
      IF(IDIAG.EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,E11.5)')
     &   '        Individual second order energy threshold ',E_THRE
        WRITE(LUOUT,'(1H ,A,E11.5)')
     &   '        Individual first order wavefunction threshold ',C_THRE
        IF(ICLSSEL.EQ.1) THEN
         WRITE(LUOUT,*)
         WRITE(LUOUT,'(1H ,A)') 
     &   '         Class selection will be performed: ' 
         WRITE(LUOUT,'(1H ,A)') 
     &   '         =================================== ' 
         WRITE(LUOUT,'(1H ,A,E11.5)')
     &    '          Total second order energy threshold ',E_CONV
         WRITE(LUOUT,'(1H ,A,E11.5)')
     &    '          Total first order wavefunction threshold ',C_CONV
        ELSE
         WRITE(LUOUT,'(1H ,A)') 
     &'            No class selection in iterative procedure '
        END IF
      END IF
      IF(I_DO_COMHAM.EQ.1) THEN
         WRITE(LUOUT,'(1H ,A)') 
     &'            Complete Hamiltonian will be constructed in CI '
      END IF
      IF(I_DO_DUMP_FOR_MRPT.EQ.1) THEN
         WRITE(LUOUT,'(1H ,A)') 
     &'            H0 and V will be dumped for use in MRPT1 program '
      END IF
C     END IF
*
      IF(IPERT.NE.0) THEN
        WRITE(LUOUT,'(1H ,A)')
     &    '     Perturbation calculation'
        WRITE(LUOUT,'(1H ,A)')
     &  '     ======================= '
        WRITE(6,*)
     &  '        Root Choosen as zero order state ', IRFROOT
        WRITE(6,*)
     &  '        Root used for zero order operator ', IH0ROOT
COLD    IF(MPORENP.EQ.1) THEN
COLD    WRITE(6,*)
COLD &  '        Moller Plesset partitioning '
COLD    ELSE IF (MPORENP.EQ.2) THEN
COLD    WRITE(6,*)
COLD &  '        Epstein-Nesbet partitioning '
COLD    ELSE IF  (MPORENP.EQ.0) THEN
COLD    WRITE(6,*)
COLD &  '        One-body Hamiltonian readin '
COLD    END IF
        IF(IE0AVEX.EQ.1) THEN
          WRITE(6,*) 
     &  '        Expectation value of H0 used as zero order energy '
        ELSE IF( IE0AVEX.EQ.2) THEN
          WRITE(6,*) 
     &  '        Exact energy of reference used as zero order energy'
        END IF
        WRITE(6,*)
     &  '        Correction vectors obtained through  order ', NPERT
        IF(IH0SPC.EQ.0) THEN
        WRITE(6,*)
     &  '        No restrictions on perturbation interactions '      
        ELSE
        WRITE(6,*) 
     &  '        Perturbation restricted to interactions in subspaces'
        END IF
*
        IF(IH0SPC.NE.0) THEN
        WRITE(6,*)
        WRITE(6,*) 
     &  '        Number of perturbation subspaces ', NPTSPC
        WRITE(6,*)
        WRITE(6,*)
     &  '        ======================== '
        WRITE(6,*) 
     &  '        Perturbation subspaces: '
        WRITE(6,*) 
     &  '        ======================== '
        DO JPTSPC = 1, NPTSPC
COLD      WRITE(LUOUT,'(A)')
COLD &     ' ====================================================== '
          WRITE(LUOUT,'(A)')
          WRITE(LUOUT,'(7X,A)') '         Min. occ    Max. occ '
          WRITE(LUOUT,'(7X,A)') '         ========    ======== '
          DO IGAS = 1, NGAS
            WRITE(LUOUT,'(7X,A,I2,3X,I3,9X,I3)')
     &      '   GAS',IGAS,IOCPTSPC(1,IGAS,JPTSPC)
     &                   ,IOCPTSPC(2,IGAS,JPTSPC)
          END DO
        END DO
*    
        WRITE(6,*) 
        WRITE(6,'(7X,A)') ' ========================================'
        WRITE(6,'(7X,A)') ' Approximate Hamiltonian in CI subspaces '
        WRITE(6,'(7X,A)') ' ========================================'
        WRITE(6,'(7X,A)') 
        WRITE(6,'(7X,A)') '    Subspace          H(apr)   '
        WRITE(6,'(7X,A)') '  ============================='
        WRITE(6,'(7X,A)')
        DO JPTSPC = 1, NPTSPC
          IF(IH0INSPC(JPTSPC).EQ.1) THEN
            WRITE(LUOUT,'(12X,I3,8X,A)')
     &      JPTSPC, ' Diagonal Fock operator '
          ELSE IF(IH0INSPC(JPTSPC).EQ.2) THEN
            WRITE(LUOUT,'(12X,I3,8X,A)')
     &      JPTSPC, ' Epstein-Nesbet operator'
          ELSE IF(IH0INSPC(JPTSPC).EQ.3) THEN
            WRITE(LUOUT,'(12X,I3,8X,A)')
     &      JPTSPC, ' Nondiagonal Fock operator '
          ELSE IF(IH0INSPC(JPTSPC).EQ.4) THEN
            WRITE(LUOUT,'(12X,I3,8X,A)')
     &      JPTSPC, ' Complete Hamiltonian  '
          ELSE IF(IH0INSPC(JPTSPC).EQ.5) THEN
            WRITE(LUOUT,'(12X,I3,8X,A)')
     &      JPTSPC, ' Mix of Fock and Exact operator '
          END IF
         END DO
         IF(ISETKW(61).GT.0) THEN
           WRITE(6,*)
           WRITE(6,'(7X,A)') 
     &     ' Orbital subspaces where exact Hamiltonian is used: '
           WRITE(6,'(7X,A)')
     &      '===================================================='
           WRITE(6,*)
           WRITE(LUOUT,'(10X,10(2X,I3))') (IH0EXSPC(I),I=1, NH0EXSPC)
           WRITE(6,*) 
         END IF
*      
       END IF
       END IF
*
       I_AM_DOING_BK = 0
       IF(I_AM_GIOVANNI.EQ.1) THEN
         WRITE(6,*) ' BK-like approximation in action'
         WRITE(6,*) ' Min and Max for subspace with exact Hamiltonian'
         WRITE(6,*) ' ==============================================='
*
         WRITE(6,'(A)')
         WRITE(6,'(A)') '         Min. occ    Max. occ '
         WRITE(6,'(A)') '         ========    ======== '
         DO IGAS = 1, NGAS
           WRITE(6,'(A,I2,3X,I3,9X,I3)')
     &     '   GAS',IGAS,IOCCPSPC(IGAS,1),IOCCPSPC(IGAS,2)
         END DO
*. For transfer to SBLOCKS
         I_AM_DOING_BK = 1
       END IF
         
*
* Coupled cluster calculations
*
       IF(I_DO_CC.NE.0) THEN
        WRITE(LUOUT,'(1H ,A)') '       Coupled cluster calculations'
        WRITE(LUOUT,'(1H ,A)') '     ================================ '
        WRITE(LUOUT,*)
        IF(CCFORM(1:3).EQ.'TCC' ) THEN
           WRITE(LUOUT,'(1H ,A)') '        Traditional CC (TCC)  '
        ELSE IF (CCFORM(1:3).EQ.'VCC') THEN
           WRITE(LUOUT,'(1H ,A)') '        Variational CC (VCC)  '
        ELSE IF (CCFORM(1:3).EQ.'UCC') THEN
           WRITE(LUOUT,'(1H ,A)') '        Unitary CC (UCC)  '
        END IF
        IF(ISPIN_RESTRICTED.EQ.1) THEN
           WRITE(LUOUT,'(1H ,A)') '        Spin-restricted calc '
        END IF
*
        IF(I_DO_ICCC.EQ.1) THEN
          IF(I_APPROX_HCOM_E.EQ.1)   WRITE(LUOUT,'(1H ,A)')  
     &    '  Approximate Hamiltonian in highest commutator for energy'
          IF(I_APPROX_HCOM_V.EQ.1)   WRITE(LUOUT,'(1H ,A)')  
     &    '  Approximate Hamiltonian in highest commutator for vecfnc'
          IF(I_APPROX_HCOM_J.EQ.1)   WRITE(LUOUT,'(1H ,A)')  
     &    '  Approximate Hamiltonian in highest commutator for Jacobian'
        END IF
*
        IF(ICCSOLVE.EQ.1) THEN
           WRITE(LUOUT,'(1H ,A)') '        Solver: simple Pert-exp '
        ELSE IF(ICCSOLVE.EQ.2) THEN
           WRITE(LUOUT,'(1H ,A)') '        Solver: DIIS            '
           WRITE(LUOUT,'(1H ,7X,A,I3)') ' Max dim of subspace =', 
     &     MAX_DIIS_VEC
        END IF
*
        IF(I_DO_SBSPJA.EQ.1) THEN
           WRITE(LUOUT,'(1H ,7X,A)') ' Subspace Jacobian used '
           WRITE(LUOUT,'(1H ,7X,A,I3)') ' Max dim of subspace = ',
     &          MAX_VEC_APRJ
           WRITE(LUOUT,'(1H ,7X,A,E8.3)')
     &          ' Max step length                   = ',XMXSTP
           WRITE(LUOUT,'(1H ,7X,A,E8.3)')
     &          ' Max subspace sampling step length = ',XMXSTP
        END IF
        I_DO_UPDIA = -2303
        IF(I_DO_UPDIA.EQ.1) THEN
           WRITE(LUOUT,'(1H ,7X,A,I3)') ' Diagonal update used '
        END IF
*
        WRITE(LUOUT,'(1H ,7X,A,E15.8)') 
     &  ' Convergence threshold for norm of vectorfuntion ', CCCONV
        IF(I_DO_CCN.EQ.1) THEN
         WRITE(LUOUT,'(1H ,A)')'         CCN Jacobiant constructed'
        END IF
*
        IF(I_DO_CC3.EQ.1) THEN
         WRITE(LUOUT,'(1H ,A)')'         CC3 approximation for triples'
         WRITE(LUOUT,'(1H ,A)')'         CC3 Jacobiant constructed'
        END IF
* 
        IF(MXSPOX.NE.0) THEN
         WRITE(LUOUT,'(1H ,7X,A,I3)')
     &   ' Largest allowed spin-orbital excitation level ',MXSPOX
        ELSE 
         WRITE(LUOUT,'(1H ,A,A)')
     &   ' Largest allowed spin-orbital excitation level = ',
     &   ' max. orbital excitation level'
        END IF
*
        IF(IUSE_TR .EQ. 1) THEN
         WRITE(LUOUT,'(1H ,7X,A)')' Time-reversal used '
        END IF
*
        IF(I_DO_CI_TO_CC.EQ.1) THEN
          WRITE(6,'(1H ,9X,A)') 
     &    ' CI to CC transformation, output on unit 93 '
        END IF
*
        IF(I_DO_CC_EXC_E.EQ.1) THEN
          WRITE(6,'(1H , 7X,A)') 
     &   ' CC excitation energies will be calculated '
          WRITE(6,*)
          WRITE(6,'(1H , 7X,A)') 
     &    ' Number of excitation energies per symmetry:'
          WRITE(6,'(12X,10I3)') (NEXC_PER_SYM(ISM),ISM=1,NIRREP)
*
          IF(IRES_EXC.EQ.1) THEN
          WRITE(6,'(1H , 7X,A)') 
     &    ' Restart in first CC excitation-calculation '
          END IF
*
        END IF
*       ^ End if CC calculation is to be carried out
*
        IF(I_DO_MASK_CC.EQ.1) THEN
           WRITE(6,*) ' Single determinant used define p/h space: '
           WRITE(6,'(1H ,20I3)') (MASK_SD(IEL,1),IEL=1, MSK_AEL)
           WRITE(6,'(1H ,20I3)') (MASK_SD(IEL,2),IEL=1, MSK_BEL)
        END IF
        IF(NOAAEX.EQ.1) THEN 
           WRITE(6,'(1H ,7X,A)')
     &  ' No pure active-active rotations '
        END IF
*
        WRITE(6,'(1H ,7X,A,I3)') 
     &  ' Dimension of resolution-strings for CC = ', MXINKA_CC
*
        IF(MSCOMB_CC.EQ.1)  WRITE(6,'(1H ,7X,A)') 
     &  ' Spincombinations used for CC operator '                     
*
        IF(ISIMTRH.EQ.1) THEN
          WRITE(6,'(1H ,8X,A)') 
     &    'Singles included through similarity transformed Hamiltonian'
        END IF
*
        IF(IFRZ_CC .EQ. 1 ) THEN
          WRITE(6,'(1H ,7X,A,12I3)') 
     &    ' Frozen CC excitation levels: ',(IFRZ_CC_AR(I),I=1,NFRZ_CC)
        END IF
*
        IF(I_DO_CC_EXP.EQ.1) THEN
          WRITE(6,'(1H ,7X,A)') 
     &    ' Expectation value of H calculated in CI space '             
        END IF
*
        IF(I_DO_NEWCCV.EQ.0) THEN
          WRITE(6,'(1H ,7X,A)') 
     &    ' CI approach used for CC vector function '
        ELSE IF(I_DO_NEWCCV.EQ.1) THEN
          WRITE(6,'(1H ,7X,A)') 
     &    ' H_EF approach used for  CC vector function    '             
        ELSE IF ( I_DO_NEWCCV.EQ.2) THEN 
          WRITE(6,'(1H ,7X,A)') 
     &    ' Commutator approach used for CC vector function  '      
        END IF
*
        IF(I_DO_NEWCCV.EQ.0) THEN
          IF(I_USE_NEWCCP.EQ.0) THEN
            WRITE(6,*) '        Old phase convention '
          ELSE IF (I_USE_NEWCCP.GE.1) THEN
            WRITE(6,*) '        New phase convention '
          END IF
        END IF
        IF(I_DO_CC_TO_CI.EQ.1) THEN
          WRITE(6,*) 
     &    '        CC to CI conversion after last CC calculation '
        END IF
*
       END IF
*      ^ End if I do CC calculations 
       IF(I_DO_HF.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)') '       Hartree-Fock Calculations      '
        WRITE(LUOUT,'(1H ,A)') '     ================================ '
        WRITE(LUOUT,*)
*
        WRITE(LUOUT,'(A,15(1X,I2))') 
     &  '        Doubly occupied irreps: ',
     &  (NHFD_IRREP_SUPSYM(I), I = 1, NACT_SUPSYM_IRREP)
*
        WRITE(LUOUT,'(A,15(1X,I2))') 
     &  '        Singly occupied irreps: ',
     &  (NHFS_IRREP_SUPSYM(I), I = 1, NACT_SUPSYM_IRREP)
*
*. Guess of initial orbitals/density
        IF(INI_HF_MO.EQ.1) THEN
          WRITE(6,*) 
     &    '       Initial MO''s obtained by diagonalization of H(one)'
        ELSE IF(INI_HF_MO.EQ.2) THEN
          WRITE(6,*) 
     &    '       Initial MO''s readin'
        END IF
*. Optimization method used 
        IF(IHFSOLVE.EQ.1) THEN
          WRITE(6,*) 
     &    '       HF wavefunction optimization: Standard Roothaan-Hall'
        ELSE IF (IHFSOLVE.EQ.2) THEN
          WRITE(6,*) 
     &    '       HF wavefunction optimization: EOPD'
        ELSE IF (IHFSOLVE.EQ.3) THEN
          WRITE(6,*) 
     &    '       HF wavefunction optimization: One-step'
        ELSE IF (IHFSOLVE.EQ.4) THEN
          WRITE(6,*) 
     &    '       HF wavefunction optimization: Second order method'
        END IF
*
       END IF
*      ^ End if HF will be used
*
       IF(NPROP.EQ.0) THEN
       WRITE(6,*)
C        WRITE(6,*) '     No calculation of properties'
       ELSE
         WRITE(6,'(7X,A,I3)')
     &   ' Number of properties to be calculated', NPROP
         WRITE(6,*)
         WRITE(6,'(9X,A)')    ' Properties: '
         WRITE(6,'(9X,A)')   ' =============' 
         DO IPROP = 1, NPROP
           WRITE(6,'(16X,A)') PROPER(IPROP)
         END DO
*
         IF(IRELAX.EQ.0) THEN
           WRITE(6,'(7X,A)') ' No use of relaxed densities '
         ELSE
           WRITE(6,'(7X,A)') 
     &     ' Relaxed densities used for property evaluation'
C          WRITE(6,'(7X,A)') ' (implemented only for pert) '
         END IF
       END IF
*       
       IF(IEXTKOP.EQ.0.AND.IPTEKT.EQ.0) THEN
C        WRITE(6,'(5X,A)') ' No extended Koopmans'' calculations '
       ELSE IF(IEXTKOP.NE.0) THEN
         WRITE(6,'(5X,A)') ' Extended Koopmans'' calculations '
       ELSE IF(IPTEKT.NE.0) THEN
         WRITE(6,'(5X,A)') ' Perturbation expansion of EKT equations'
       END IF
*
       IF(IPTFOCK.EQ.1) THEN
         WRITE(6,*) ' Perturbation expansion of Fock matrix '
       ELSE
C        WRITE(6,*) 'No  Perturbation expansion of Fock matrix '
       END IF
*
      IF(ITRAPRP.EQ.0) THEN
C       WRITE(6,*)
C       WRITE(6,'(5X,A)') 
C    &  ' No transition properties will be calculated'
      ELSE
        WRITE(6,*)
        WRITE(6,'(5X,A)') 
     &  ' Transition properties will be calculated '
        WRITE(6,*)  ' Symmetry of additional states:', IEXCSYM
        WRITE(6,*)  ' Number   of additional states:', NEXCSTATE
        WRITE(6,*)
      END IF
*
      IF(I_DO_LZ2.EQ.1) THEN
        WRITE(6,*) ' The expectation value of Lz2 will be calculated '
      END IF
*
      IF(IGENTRD.EQ.1) THEN
        WRITE(6,*) ' General transition density will be calculated' 
        WRITE(6,*) ' between last state specified in GASOCC and: '
        WRITE(6,*) ' state with: '
        WRITE(6,*)
        WRITE(6,*) '   Symmetry ', IGST_SM
        WRITE(6,*) '   2 * Ms   ', IGST_MS2
        WRITE(6,*) '   occupation constraints: '
        WRITE(6,'(5X,20I3)') ((IGST_OCC(ISPC,IMAXMIN),IMAXMIN=1,2),
     &                        ISPC = 1, NGAS)
      END IF
*
      IF(IRESPONS.NE.0) THEN
      WRITE(LUOUT,*)
      WRITE(LUOUT,*) '**************************'
      WRITE(LUOUT,*) '*  Response Calculation  *'
      WRITE(LUOUT,*) '************************** '
      WRITE(LUOUT,*)
        WRITE(6,*)  
     &  ' CI-Response will be called after each CI calculation'
        WRITE(6,*) 
     &  ' Root used for response calculations (RFROOT) ',IRFROOT
        WRITE(6,*)
C       WRITE(6,*) ' Number of A-operators: ', N_AVE_OP  
        WRITE(6,*) ' Labels of A-operators '
        WRITE(6,*) ' ======================='
        WRITE(6,*)
        DO IAVE = 1, N_AVE_OP
          WRITE(6,'(1H , 6X,A)') AVE_OP(IAVE)
        END DO
        WRITE(6,*)
C       WRITE(6,*) ' Number of response calculations ', NRESP
        WRITE(6,*) ' Perturbations: '
        WRITE(6,*) ' ================'
        WRITE(6,*)
        WRITE(6,*) ' Calc  Op1    Op2    Mxord1     Mxord2    Freq '
        DO IRESP = 1, NRESP
          WRITE(6,'(1H ,I2,2X,A,A,3X,I4,3X,I4,2X,F12.7)' )
     &    IRESP,RESP_OP(1,IRESP),RESP_OP(2,IRESP),MAXORD_OP(1,IRESP),
     &    MAXORD_OP(2,IRESP),RESP_W(IRESP)
        END DO
      END IF
*
C     IF(NOMOFL.EQ.0) THEN
        WRITE(6,*)
        WRITE(6,'(7X,A)') ' Final orbitals:'
        WRITE(6,'(7X,A)') ' ==============='
        WRITE(6,*)
*
        IF(I_USE_SUPSYM.EQ.1) THEN
          IF(I_NEGLECT_SUPSYM_FINAL_MO .EQ. 1) THEN
           WRITE(6,'(10X,A)')
     &     ' Orbital will be in occupation supersymmetry  order'
          ELSE
           WRITE(6,'(10X,A)')
     &     ' Orbital will be in standard supersymmetry  order'
          END IF
        END IF
*
        IF(IFINMO.EQ.0) THEN
          WRITE(6,'(10X,A)') ' No additional rotations'   
        ELSE IF(IFINMO.EQ.1) THEN
          WRITE(6,'(10X,A)') ' Natural orbitals'   
        ELSE IF(IFINMO.EQ.2) THEN
          WRITE(6,'(10X,A)') ' Canonical orbitals'   
        ELSE IF(IFINMO.EQ.3) THEN
          WRITE(6,'(10X,A)') ' Pseudo-natural orbitals'   
          WRITE(6,'(10X,A)') 
     &   ' (Density matrix diagonalized in orbital subspaces )'
        ELSE IF(IFINMO.EQ.4) THEN
          WRITE(6,'(10X,A)') ' Pseudo-canonical orbitals'   
          WRITE(6,'(10X,A)') 
     &   ' (FI+FA  diagonalized in orbital subspaces )'
         ELSE IF (IFINMO .EQ. 5 ) THEN
          WRITE(6,'(10X,A)') 
     &   ' Pseudo-natural-canonical orbitals (sic)'
          WRITE(6,'(10X,A)') 
     &   ' (Pseudo natural orbitals are first obtained'
          WRITE(6,'(10X,A)') 
     &   '  by diagonalizing density matrix in orbital subpspaces.'
          WRITE(6,'(10X,A)') 
     &   '  FI+FA is transformed to this basis, and the transformed'
          WRITE(6,'(10X,A)') 
     &   '  matrix is block diagonalized) '                          
          WRITE(6,*)
          WRITE(6,'(10X,A)') 
     &   ' Orbital spaces in which transformed FIFA is diagonalized'
          WRITE(6,'(10X,A)') 
     &   ' ========================================================'
          DO IPSSPC = 1, NPSSPC
            WRITE(LUOUT,'(A,I2,A,10I4,6X,2I6)')
     &      '     SPACE',IPSSPC,'          ', 
     &     (NPSSH(IRREP,IPSSPC),IRREP = 1, NIRREP) 
          END DO
        END IF
C     END IF
*. Transformation of CI vectors
      IF(ITRACI.EQ.0) THEN
C       WRITE(6,'(5X,A)')  ' No transformation of CI vectors'
      ELSE
        WRITE(6,'(5X,A)')   ' CI vectors transformed in each run'
        WRITE(6,'(7X,A,A)') 
     &        ' Complete or restricted rotations:',ITRACI_CR
        WRITE(6,'(7X,A,A)') 
     &        ' Type of Final orbitals          :',ITRACI_CN
      END IF
*
* Integral Transformations 
*
      WRITE(6,*)
      WRITE(6,*) ' Storage and transformation of integrals '
      WRITE(6,*) ' ======================================= '
      WRITE(6,*)
      IF(ITRA_ROUTE.EQ.1) THEN
        WRITE(6,*) ' Old form in use '
      ELSE
        WRITE(6,*) ' New form in use '
      END IF
*
      IF(ITRA_FI.EQ.1) THEN
        WRITE(6,*) '      Integrals transformed to final MO''s '
      END IF
      IF(ITRA_IN.EQ.1) THEN
        WRITE(6,*) '      Integrals transformed to initial  MO''s '
      END IF
*
*. Reorder orbitals ?
*
      IF(I_DO_REO_ORB.EQ.1) THEN    
        WRITE(6,*) ' Orbitals will be reordered '
        WRITE(6,*) ' ==========================='
        WRITE(6,*)
        WRITE(6,*) ' Symmetry  Old number   New number '
        WRITE(6,*) ' =================================='
        WRITE(6,*)
        DO ISWITCH = 1, NSWITCH
          WRITE(6,'(3(2X,I6))') 
     &    IREO_ORB(1,ISWITCH), IREO_ORB(2,ISWITCH),IREO_ORB(3,ISWITCH)
        END DO
      END IF
*
* Fragments defining molecule
*
      IF(NFRAG_TP.NE.0) THEN
        WRITE(6,*)
        WRITE(6,*) ' Molecule is defined in terms of fragments'
        WRITE(6,*) ' ========================================='
        WRITE(6,*) 
        WRITE(6,*) ' Number of fragment types:', NFRAG_TP
        WRITE(6,*) ' Number of fragments in molecule ', 
     &             NFRAG_MOL
        WRITE(6,*)
        WRITE(6,*) ' Character strings of fragments: '
        DO IFRAG = 1, NFRAG_TP
          WRITE(6,'(2X,A3)') CFRAG(IFRAG)
        END DO
        WRITE(6,*) ' Molecule in terms of fragment: '
        WRITE(6,'(40 A3)') 
     &  (CFRAG(IFRAG_MOL(IFRAG)),IFRAG = 1, NFRAG_MOL)
*
        WRITE(6,*)   
        WRITE(6,*) ' Division of fragments into equivalent groups: '
        WRITE(6,*)   
        WRITE(6,*) ' Equivalent group, Dimension, Fragments '
        WRITE(6,*) ' ======================================='
        DO IEQV = 1, NEQVGRP_FRAG
          WRITE(6,'(2X,I2,3X,I2,10(1X,I2))')
     &    IEQV, LEQVGRP_FRAG(IEQV), 
     &    (IEQVGRP_FRAG(IFRAG,IEQV),IFRAG=1,LEQVGRP_FRAG(IEQV))
        END DO
*
      END IF
*
      WRITE(6,'(A)') ' Initial set of orbitals '
      WRITE(6,'(A)') ' ======================= '
*. Form of initial orbitals
      IF(INI_MO_TP.EQ.1) THEN
        WRITE(6,'(5X,A)')  ' Atomic orbitals '
      ELSE IF (INI_MO_TP.EQ.3) THEN
        WRITE(6,'(5X,A)')  ' Read in from Environment '
      ELSE IF (INI_MO_TP.EQ.2) THEN
        WRITE(6,'(5X,A)')  ' Read in from Environment and modified'
       ELSE IF(INI_MO_TP.EQ.4) THEN
        WRITE(6,'(5X,A)')  ' Built from fragment orbitals '
       ELSE IF(INI_MO_TP.EQ.5) THEN
        WRITE(6,'(5X,A)')  ' Read in from LUCINF_O and orthonormalized'
       END IF
*
       IF(INI_MO_ORT.EQ.1) THEN
        WRITE(6,'(5X,A)') ' Symmetric orthogonalization '
       ELSE
        WRITE(6,'(5X,A)') ' Orthogonalization by diagonalization '
       END IF
*
       IF(I_DO_NORTCI.EQ.1) THEN
        IF(INI_ORT_VBGAS.EQ.0) THEN
          WRITE(6,'(5X,A)') ' VB gaspace will not be orthogonalized'
        ELSE
          WRITE(6,'(5X,A)') ' VB gaspace will be orthogonalized'
        END IF
       END IF
     
        
*
*
*. Print levels
*
      WRITE(LUOUT,*)
      WRITE(LUOUT,'(1H ,A)')  '     Print levels: '
      IF(ISETKW(29).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for string    information = ', IPRSTR
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for string    information = ', IPRSTR
      END IF
      IF(ISETKW(30).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for CI space  information = ', IPRCIX
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for CI space  information = ', IPRCIX
      END IF
      IF(ISETKW(31).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for orbital   information = ', IPRORB
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for orbital   information = ', IPRORB
      END IF
      IF(ISETKW(65).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for density matrix        = ', IPRDEN
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for density matrix        = ', IPRDEN
      END IF
      IF(ISETKW(32).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for iterative information = ', IPRDIA
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for iterative information = ', IPRDIA
      END IF
      IF(ISETKW(159).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for CSF information       = ', IPRCSF
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for CSF information       = ', IPRCSF
      END IF
*
      IF(ISETKW(169).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for MCSCF information     = ', 
     &  IPRMCSCF
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for MCSCF information     = ', 
     &  IPRMCSCF
      END IF
*
      IF(ISETKW(179).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for VB    information     = ', 
     &  IPRVB   
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Changed print level for VB    information     = ', 
     &  IPRVB   
      END IF
      IF(ISETKW(181).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for integrals             = ', 
     &  IPRINTEGRAL
      ELSE
        WRITE(LUOUT,'(1H ,A,I4)')
     &  '      Changed print level for Integral information  = ', 
     &  IPRINTEGRAL
      END IF
*
*
      IF(NPROP.NE.0) THEN
        IF(ISETKW(99).EQ.2) THEN
          WRITE(6,'(1H ,A,I3)') 
     &  '      Default print level for properties            = ', IPRPRO
        ELSE
          WRITE(6,'(1H ,A,I3)') 
     &  '      Changed print level for properties            = ', IPRPRO
        END IF
*
      END IF
*
      IF(IRESPONS.NE.0) THEN
      IF(ISETKW(84).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for response section      = ', IPRRSP
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Raised  print level for response section      = ', IPRRSP
      END IF
      END IF
*
      IF(I_DO_CC.EQ.1) THEN
      IF(ISETKW(84).EQ.2) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Default print level for coupled cluster       = ', IPRCC
      ELSE
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Raised  print level for  coupled cluster      = ', IPRCC
      END IF
      END IF
*
      IF(IPROCC.NE.0) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Occupation of SD''s/ configurations           = ',IPROCC
      END IF
*
      IF(IPRNCIV.EQ.1 ) THEN
        WRITE(LUOUT,'(1H ,A,I3)')
     &  '      Final CI vectors will be printed '
      END IF
*
      WRITE(6,*)
C?    IF(MOLCS.EQ.1) WRITE(LUOUT,'(1H ,A,E18.9)') 
C?   &  '      Core energy: ', ECORE
*
      IF(IDMPIN.EQ.1) THEN
        WRITE(LUOUT,'(1H ,A)')
        WRITE(6,*) '      Integrals written in formatted form (E22.15)'
        WRITE(6,*) '      on file 90 '
      END IF
*
C?    WRITE(6,*) ' IPART before leaving READIN = ', IPART
      RETURN
      END
      SUBROUTINE RESID(LUC,LUHC,LUOUT,LUDIA,LBLK,
     &                VEC1,VEC2,VEC3,NSUB,ISUB,SUBTRM,SCALE,ENOT,EP)
*
* SAVE SCALE*( C - (H0 - ENOT)**-1(HC - EP*C )) ON DISC
*
* ELEMENTS CORRESPONDIND TO THE EXPLICIT SUBSPACE
* ARE READ FROM SUBTRM
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION VEC1(*),VEC2(*),VEC3(*)
      DIMENSION ISUB(*),SUBTRM(*)
*
C?    WRITE(6,*) ' RESID SPEAKING '
C?    WRITE(6,*) '================'
      IBASE = 1
      IEFF = 1
      CALL REWINE(LUC,LBLK)
      CALL REWINE(LUHC,LBLK)
      CALL REWINE(LUDIA,LBLK)
      CALL REWINE(LUOUT,LBLK)
*
*. LOOP OVER RECORDS OF VECTORS
 1000 CONTINUE
* RECORD OF C IN VEC1,HCI IN VEC2, HNOT IN VEC3
C                NEXREC(LU,LBLK,REC,IEND,LENGTH)
        CALL NEXREC(LUC,LBLK,VEC1,IEND,LENGTH)
        CALL NEXREC(LUHC,LBLK,VEC2,IEND,LENGTH)
        CALL NEXREC(LUDIA,LBLK,VEC3,IEND,LENGTH)
C?      WRITE(6,*) ' NEXT SEGMENT OF VECTOR,C,HC,H0 '
C?      CALL WRTMAT(VEC1,1,LENGTH,1,LENGTH)
C?      CALL WRTMAT(VEC2,1,LENGTH,1,LENGTH)
C?      CALL WRTMAT(VEC3,1,LENGTH,1,LENGTH)
 
 
        IF(IEND .EQ. 0 ) THEN
*.HC - EP*C OVERWRITES HC
          CALL VECSUM(VEC2,VEC2,VEC1,1.0D0,-EP,LENGTH)
*.(HNOT-ENOT) ** -1 (HC-EP*C) OVERWRITES HC
C     DIAVC3(VECOUT,VECIN,DIAG,SHIFT,NDIM,VDSV)
          CALL DIAVC3(VEC2,VEC2,VEC3,-ENOT,LENGTH,VDSV)
C$        CCC = INPROD(VEC2,VEC1,LENGTH)
C$        WRITE(6,*) ' CCC IN RESID ',CCC
 
*.C -(HNOT-ENOT)**-1(HC-EP*C) OVERWRITES VEC2
          CALL VECSUM(VEC2,VEC1,VEC2,1.0D0,-1.0D0,LENGTH)
*. SCATTER SUBSPACE ELEMENTS OUT
          IF(IEFF.LE.NSUB) THEN
   10       CONTINUE
            IF(ISUB(IEFF).GE.IBASE.AND.ISUB(IEFF).LE.IBASE+LENGTH-1)
     &      THEN
              VEC2(ISUB(IEFF)-IBASE+1) = SUBTRM(IEFF)
              IEFF = IEFF + 1
              IF(IEFF.LE.NSUB) GOTO 10
            END IF
          END IF
          CALL SCALVE(VEC2,SCALE,LENGTH)
C?        WRITE(6,*) ' NEW C SEGMENT '
C?        CALL WRTMAT(VEC2,1,LENGTH,1,LENGTH)
          CALL PUTREC(LUOUT,LBLK,VEC2,LENGTH)
C              PUTREC(LU,LBLK,REC,LENGTH)
         IBASE = IBASE + LENGTH
      GOTO 1000
        END IF
*
      IF(LBLK.LE.0) CALL ITODS(-1,1,LBLK,LUOUT)
*
      RETURN
      END
      SUBROUTINE RSMXMN(MAXEL,MINEL,NORB1,NORB2,NORB3,NEL,
     &                  MIN1,MAX1,MIN3,MAX3,NTEST)
*
* Construct accumulated MAX and MIN arrays for a RAS set of strings
*
      IMPLICIT REAL*8           ( A-H,O-Z)
      DIMENSION  MINEL(*),MAXEL(*)
*
      NORB = NORB1 + NORB2 + NORB3
*. accumulated max and min in each of the three spaces
*. ( required max and min at final orbital in each space )
COLD  MIN1A = MIN1
      MIN1A = MAX(MIN1,NEL-MAX3-NORB2)
      MAX1A = MAX1
*
      MIN2A = NEL - MAX3
      MAX2A = NEL - MIN3
*
      MIN3A = NEL
      MAX3A = NEL
*
      DO 100 IORB = 1, NORB
        IF(IORB .LE. NORB1 ) THEN
          MINEL(IORB) = MAX(MIN1A+IORB-NORB1,0)
          MAXEL(IORB) = MIN(IORB,MAX1A)
        ELSE IF ( NORB1.LT.IORB .AND. IORB.LE.(NORB1+NORB2)) THEN
          MINEL(IORB) = MAX(MIN2A+IORB-NORB1-NORB2,0)
          IF(NORB1 .GT. 0 )
     &    MINEL(IORB) = MAX(MINEL(IORB),MINEL(NORB1))
          MAXEL(IORB) = MIN(IORB,MAX2A)
        ELSE IF ( IORB .GT. NORB1 + NORB2 ) THEN
          MINEL(IORB) = MAX(MIN3A+IORB-NORB,0)
          IF(NORB1+NORB2 .GT. 0 )
     &    MINEL(IORB) = MAX(MINEL(IORB),MINEL(NORB1+NORB2))
          MAXEL(IORB) = MIN(IORB,MAX3A)
        END IF
  100 CONTINUE
*
      IF( NTEST .GE. 100 ) THEN
        WRITE(6,*) ' Output from RSMXMN '
        WRITE(6,*) ' ================== '
        WRITE(6,*) ' MINEL: '
        CALL IWRTMA(MINEL,1,NORB,1,NORB)
        WRITE(6,*) ' MAXEL: '
        CALL IWRTMA(MAXEL,1,NORB,1,NORB)
      END IF
*
      RETURN
      END
C                 CALL RSSBCB(IASM,IATP,IOCTPA,
C    &                 IBSM,IBTP,IOCTPB,
C    &                 LLASM,LLATP,LLBSM,LLBTP,NGAS,
C    &                 NELFSPGP(1,IATP+IOCTPA-1),
C    &                 NELFSPGP(1,IBTP+IOCTPB-1),
C    &                 NAEL,NBEL,
C    &                 IAGRP,IBGRP,
C    &                 SB(ISOFF),CB(ICOFF),IDOH2,
C    &                 ADSXA,SXSTST,STSTSX,DXSTST,STSTDX,SXDXSX,
C    &                 NTSOB,IBTSOB,ITSOB,MAXI,MAXK,
C    &                 SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,C2,
C    &                 NSMOB,NSMST,NSMSX,NSMDX,
C    &                 NIA,NIB,NLLA,NLLB,MXPOBS,IDC,PS,
C    &                 ICJKAIB,CJRES,SIRES,I3,XI3S,I4,XI4S,
C    &                 MXSXBL,MXSXST,MOCAA,MOCAB,IAPR,IPRNT)
      SUBROUTINE RSSBCB(IASM,IATP,IOCPTA,
     &                  IBSM,IBTP,IOCTPB,
     &                  JASM,JATP,JBSM,JBTP,NGAS,
     &                  IAOC,IBOC,JAOC,JBOC, 
     &                  NAEL,NBEL,
     &                  IJAGRP,IJBGRP,
     &                  SB,CB,IDOH2,
     &                  ADSXA,SXSTST,STSTSX,DXSTST,STSTDX,SXDXSX,
     &                  NOBPTS,IOBPTS,MXPNGAS,ITSOB,MAXI,MAXK,
     &                  SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,C2,
     &                  NSMOB,NSMST,NSMSX,NSMDX,
     &                  NIA,NIB,NJA,NJB,MXPOBS,IDC,PS,
     &                  ICJKAIB,CJRES,SIRES,I3,XI3S,I4,XI4S,
     &                  MXSXBL,MXSXST,MOCAA,MOCAB,IAPR,SCLFAC,IPRNT)
*
* Contributions to sigma block (iasm iatp, ibsm ibtp ) from
* C block (jasm jatp , jbsm, jbtp)
*
* =====
* Input
* =====
*
* IASM,IATP : Symmetry and type of alpha strings in sigma
* IBSM,IBTP : Symmetry and type of beta  strings in sigma
* JASM,JATP : Symmetry and type of alpha strings in C
* JBSM,JBTP : Symmetry and type of beta  strings in C
* NGAS      : Number of active spaces in calculation
* IAOC,IBOC : Number of electrons in each AS for sigma supergroups
* JAOC,JBOC : Number of electrons in each AS for C     supergroups
* NAEL : Number of alpha electrons
* NBEL : Number of  beta electrons
* IJAGRP    : IA and JA belongs to this group of strings
* IJBGRP    : IB and JB belongs to this group of strings
* CB : Input c block
* IDOH2 : = 0 => no two electron operator
* IDOH2 : = 1 =>    two electron operator
* ADASX : sym of a+, a => sym of a+a
* ADSXA : sym of a+, a+a => sym of a
* SXSTST : Sym of sx,!st> => sym of sx !st>
* STSTSX : Sym of !st>,sx!st'> => sym of sx so <st!sx!st'>
*          is nonvanishing by symmetry
* DXSTST : Sym of dx,!st> => sym of dx !st>
* STSTDX : Sym of !st>,dx!st'> => sym of dx so <st!dx!st'>
*          is nonvanishing by symmetry
* NTSOB  : Number of orbitals per type and symmetry
* IBTSOB : base for orbitals of given type and symmetry
* IBORB  : Orbitals of given type and symmetry
* MAXI   : Largest Number of ' spectator strings 'treated simultaneously
* MAXK   : Largest number of inner resolution strings treated at simult.
*
* ICJKAIB =1 =>  construct C(Ka,Jb,j) and S(Ka,Ib,i) as intermediate 
*                 matrices in order to reduce overhead
*
* ======
* Output
* ======
* SB : fresh sigma block
*
* =======
* Scratch
* =======
* SSCR, CSCR : at least MAXIJ*MAXI*MAXK, where MAXIJ is the
*              largest number of orbital pairs of given symmetries and
*              types.
* I1, XI1S   : at least MXSTSO : Largest number of strings of given
*              type and symmetry
* I1, XI1S   : at least MXSTSO : Largest number of strings of given
*              type and symmetry
* C2 : Must hold largest STT block of sigma or C
*
* XINT : Scratch space for integrals.
*
* Jeppe Olsen , Winter of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER  ADSXA,SXSTST,STSTSX,DXSTST,STSTDX,SXDXSX
*. Output
      DIMENSION CB(*),SB(*)
*. Scratch
      DIMENSION SSCR(*),CSCR(*),I1(*),XI1S(*),I2(*),XI2S(*)
      DIMENSION I3(*),XI3S(*)
      DIMENSION C2(*)
      DIMENSION CJRES(*),SIRES(*)
*
      NTEST = 0000
      NTEST = MAX(NTEST,IPRNT)
      NTESTO= NTEST
*
C?    WRITE(6,*) ' Memcheck entering RSSBCB '
C?    CALL MEMCHK
      IF(NTEST.GE.200) THEN
        WRITE(6,*) ' =================='
        WRITE(6,*) ' RSSBCB:  C block '
        WRITE(6,*) ' ==================='
        IF(ICJKAIB.EQ.0) THEN
        CALL WRTMAT(CB,NJA,NJB,NJA,NJB)
        ELSE
        CALL WRTMAT(CB,NJB,NJA,NJB,NJA)
        END IF
        WRITE(6,*) ' =========================='
        WRITE(6,*) ' RSSBCB: Initial  S block '
        WRITE(6,*) ' =========================='
        CALL WRTMAT(SB,NIA,NIB,NIA,NIB)
*
        WRITE(6,*) ' IAOC and IBOC '
        CALL IWRTMA(IAOC,1,NGAS,1,NGAS)
        CALL IWRTMA(IBOC,1,NGAS,1,NGAS)
        WRITE(6,*) ' JAOC and JBOC  : '
        CALL IWRTMA(JAOC,1,NGAS,1,NGAS)
        CALL IWRTMA(JBOC,1,NGAS,1,NGAS)
*
        WRITE(6,*) ' IAPR and IDOH2 ', IAPR,IDOH2
  
      END IF
* Should the corresponding Hamiltonian matrix block be 
* calculated exactly or approximately
      IF(IAPR.NE.0) THEN
        STOP ' Update call to HMATAPR '
        CALL HMATAPR(IASM,IATP,IBSM,IBTP,JASM,JATP,JBSM,JBTP,
     &       IAEL1,IBEL1,IAEL3,IBEL3,JAEL1,JBEL1,JAEL3,JBEL3,
     &       IAPRLEV)
      ELSE
        IAPRLEV =-1    
      END IF
      IF(NTEST.GE.100) WRITE(6,*) ' IAPR,IAPRLEV=',IAPR,IAPRLEV
*. IAPRLEV = -1 => No approximation
*. IAPRLEV = 0  => set block to zero
*. IAPRLEV = 1  => diagonal approximation
*. IAPRLEV = 2  => Use effective one-electronoperator
*
      IF(IDC.EQ.2.AND.IATP.EQ.IBTP.AND.IASM.EQ.IBSM .AND.
     &            JASM.EQ.JBSM.AND.JATP.EQ.JBTP) THEN
*. Diagonal sigma block, use alpha-beta symmetry to reduce computations.
        IUSEAB = 1
      ELSE
        IUSEAB = 0
      END IF
*
      IF(IAPRLEV.EQ.-1) THEN
*
* Calculate block exactly 
*
      IF(IUSEAB.EQ.0.AND.IATP.EQ.JATP.AND.JASM.EQ.IASM) THEN
*
* =============================
* Sigma beta beta contribution
* =============================
*
* Sigma aa(IA,IB) = sum(i.gt.k,j.gt.l)<IB!Eb(ij)Eb(kl)!JB>
*                 * ((ij!kl)-(il!kj)) C(IA,JB)
*                 + sum(ij) <IB!Eb(ij)!JB> H(ij) C(IA,JB)
*.One electron part
*. If ICJKAIB is active matrices are transposed, so back transpose
        IF(ICJKAIB.NE.0) THEN
          CALL TRPMT3(SB,NIB,NIA,C2)
          CALL COPVEC(C2,SB,NIA*NIB)
          CALL TRPMT3(CB,NJB,NJA,C2)
          CALL COPVEC(C2,CB,NJA*NJB)
        END IF
   
        IF(NBEL.GE.1) THEN
           IF(NTEST.GE.500) THEN
             WRITE(6,*) ' SB before RSBB1E'
             call wrtmat(sb,nia,nib,nia,nib)
           END IF
          IF(NTEST.GE.101)
     &    WRITE(6,*) ' I am going to call RSBB1E'
          CALL RSBB1E(IBSM,IBTP,IOCTPB,JBSM,JBTP,IOCTPB,
     &         IJBGRP,NIA,
     &         NGAS,IBOC,JBOC,
     &         SB,CB,
     &         ADSXA,SXSTST,STSTSX,
     &         MXPNGAS,NOBPTS,IOBPTS,
     &         ITSOB,MAXI,MAXK,
     &         SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,
     &         NSMOB,NSMST,NSMSX,MXPOBS,MOCAA,
     &         NIB,MXSXST,CJRES(1),CJRES(1+MXSXBL),MOCAA,SCLFAC,NTEST)
           IF(NTEST.GE.101) THEN
             WRITE(6,*) ' SB after RSBB1E, first element '
             call wrtmat(sb,1,1    ,nia,nib)
           END IF
           IF(NTEST.GE.500) THEN
             WRITE(6,*) ' SB after RSBB1E'
             call wrtmat(sb,nia,nib,nia,nib)
           END IF
        END IF
        IF(IDOH2.NE.0.AND.NBEL.GE.2) THEN
*. Two electron part
          IF(NTEST.GE.101)
     &    WRITE(6,*) ' I am going to call RSBB2A'
          CALL RSBB2A(IBSM,IBTP,JBSM,JBTP,IJBGRP,NIA,NIB,
     &                NGAS,IBOC,JBOC,                
     &                SB,CB,
     &                ADSXA,DXSTST,STSTDX,SXDXSX,MXPNGAS,
     &                NOBPTS,IOBPTS,ITSOB,MAXI,MAXK,
     &                SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,
     &                NSMOB,NSMST,NSMSX,NSMDX,MXPOBS,
     &                CJRES,SIRES,MXSXST,MXSXBL,MOCAA,SCLFAC,NTEST,0,0)
           IF(NTEST.GE.101) THEN
             WRITE(6,*) ' SB after RSBB2A, first element '
             call wrtmat(sb,1,1    ,nia,nib)
           END IF
           IF(NTEST.GE.500) THEN
             WRITE(6,*) ' SB after RSBB2a'
             call wrtmat(sb,nia,nib,nia,nib)
           END IF
        END IF
*. If ICJKAIB is active matrices are transposed, so back transpose
        IF(ICJKAIB.NE.0) THEN
          CALL TRPMT3(SB,NIA,NIB,C2)
          CALL COPVEC(C2,SB,NIA*NIB)
          CALL TRPMT3(CB,NJA,NJB,C2)
          CALL COPVEC(C2,CB,NJA*NJB)
        END IF
      END IF
*
* =============================
* Sigma alpha beta contribution
* =============================
*
      IF(IDOH2.NE.0.AND.NAEL.GE.1.AND.NBEL.GE.1) THEN
        IF(NTEST.GE.101)
     &  WRITE(6,*) ' I am going to call RSBB2B'
        IIITRNS = 1
        IF(IIITRNS.EQ.1.AND.NIB.GT.3*NIA.AND.NJB.GT.2*NJA) THEN
           JJJTRNS = 1
        ELSE
           JJJTRNS = 0
        END IF
*
        IF (JJJTRNS.EQ.0) THEN
          CALL RSBB2B(IASM,IATP,IBSM,IBTP,NIA,NIB,
     &                JASM,JATP,JBSM,JBTP,NJA,NJB,
     &                IJAGRP,IJBGRP,NGAS,
     &                IAOC,IBOC,JAOC,JBOC,
     &                SB,CB,
     &                ADSXA,STSTSX,MXPNGAS,
     &                NOBPTS,IOBPTS,ITSOB,MAXK,
     &                SSCR,CSCR,I1,XI1S,I2,XI2S,I3,XI3S,I4,XI4S,
     &                XINT,NSMOB,NSMST,NSMSX,NSMDX,MXPOBS,
     &                IUSEAB,ICJKAIB,CJRES,SIRES,C2,SCLFAC,NTEST,0,0)
*
         ELSE IF ( JJJTRNS.EQ.1) THEN
*. well lets give the transpose routine some more practice: Transpose back
          CALL TRPMT3(SB,NIB,NIA,C2)
          CALL COPVEC(C2,SB,NIA*NIB)
*
          CALL TRPMT3(CB,NJB,NJA,C2)
          CALL COPVEC(C2,CB,NJA*NJB)
*
          CALL RSBB2B(IBSM,IBTP,IASM,IATP,NIB,NIA,
     &                JBSM,JBTP,JASM,JATP,NJB,NJA,
     &                IJBGRP,IJAGRP,NGAS,
     &                IBOC,IAOC,JBOC,JAOC,
     &                SB,CB,
     &                ADSXA,STSTSX,MXPNGAS,
     &                NOBPTS,IOBPTS,ITSOB,MAXK,
     &                SSCR,CSCR,I1,XI1S,I2,XI2S,I3,XI3S,I4,XI4S,
     &                XINT,NSMOB,NSMST,NSMSX,NSMDX,MXPOBS,
     &                IUSEAB,ICJKAIB,CJRES,SIRES,C2,SCLFAC,NTEST,0,0)

*. Transpose ( To compensate later transposition )
          CALL TRPMT3(SB,NIA,NIB,C2)
          CALL COPVEC(C2,SB,NIA*NIB)
          CALL TRPMT3(CB,NJA,NJB,C2)
          CALL COPVEC(C2,CB,NJA*NJB)
        END IF
           IF(NTEST.GE.101) THEN
             WRITE(6,*) ' SB after RSBB2B, first element '
             call wrtmat(sb,1,1    ,nia,nib)
           END IF
           IF(NTEST.GE.500) THEN
             WRITE(6,*) ' SB after RSBB2b'
             call wrtmat(sb,nia,nib,nia,nib)
           END IF
      END IF
*
* =============================
* Sigma alpha alpha contribution
* =============================
*
C     IF(IUSEAB.EQ.0) THEN
      IF(NAEL.GE.1.AND.IBTP.EQ.JBTP.AND.IBSM.EQ.JBSM) THEN
*
* alpha single excitation
*
        IF(NTEST.GE.101)
     &  WRITE(6,*) ' I am going to call RSBB1E (last time )'
        CALL RSBB1E(IASM,IATP,IOCTPA,JASM,JATP,IOCTPA,
     &                   IJAGRP,NIB,
     &                   NGAS,IAOC,JAOC,
     &                   SB,CB,
     &                   ADSXA,SXSTST,STSTSX,
     &                   MXPNGAS,NOBPTS,IOBPTS,
     &                   ITSOB,MAXI,MAXK,
     &                   SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,
     &                   NSMOB,NSMST,NSMSX,MXPOBS,MOCAA,
     &                   NIA,MXSXST,CJRES(1),CJRES(1+MXSXBL),
     &                   MOCAA,SCLFAC,NTEST)
           IF(NTEST.GE.101) THEN
             WRITE(6,*) ' SB transposed after RSBB1, first element '
             call wrtmat(sb,1,1    ,nia,nib)
           END IF
           IF(NTEST.GE.500) THEN
             WRITE(6,*) ' SB transposed  after RSBB1E'
             call wrtmat(SB,nib,nia,nib,nia)
           END IF
*
* alpha double excitation
*
        IF(IDOH2.NE.0.AND.NAEL.GE.2) THEN
          IF(NTEST.GE.101)
     &    WRITE(6,*) ' I am going to call RSBB2A (last time )'
          CALL RSBB2A(IASM,IATP,JASM,JATP,IJAGRP,NIB,NIA,
     &         NGAS,IAOC,JAOC,  
     &         SB,CB,
     &         ADSXA,DXSTST,STSTDX,SXDXSX,MXPNGAS,
     &         NOBPTS,IOBPTS,ITSOB,MAXI,MAXK,
     &         SSCR,CSCR,I1,XI1S,I2,XI2S,XINT,
     &         NSMOB,NSMST,NSMSX,NSMDX,MXPOBS,
     &         CJRES,SIRES,MXSXST,MXSXBL,MOCAA,SCLFAC,NTEST,0,0)
        END IF
*
           IF(NTEST.GE.101) THEN
             WRITE(6,*) ' SB transposed after RSBB2A, first element '
             call wrtmat(sb,1,1    ,nia,nib)
           END IF
        IF(NTEST.GE.500) THEN
          WRITE(6,*) ' SB after RSBB2A'
          call wrtmat(sb,nia,nib,nia,nib)
        END IF
      END IF
COLD  END IF
*
      ELSE IF (IAPRLEV.EQ.1) THEN
*. Approximate block with diagonal
       IFULL = 1
       CALL DIABLK(IASM,IATP,IBSM,IBTP,IFULL,C2)
       CALL VVTOV(CB,C2,C2,NIA*NIB)
       ONE = 1.0D0
*. what to do when IUSEAB = 1 is in use ???
       IF(IUSEAB.EQ.0) THEN
         FACTOR = 1.0D0
       ELSE
         FACTOR = 0.5D0
       END IF
       CALL VECSUM(SB,SB,C2,ONE,FACTOR,NIA*NIB)
      END IF
*
      IF(NTEST.GE.200) THEN
        WRITE(6,*) ' =========================='
        WRITE(6,*) ' RSSBCB: Final S block '
        WRITE(6,*) ' =========================='
        CALL WRTMAT(SB,NIA,NIB,NIA,NIB)
      END IF
      NTESTO = NTEST
C?    STOP ' Jeppe forced me to stop in RSSBCB '
      RETURN
      END
      SUBROUTINE SCAAD2(ISCA,SSCA,VECIN,VECOUT,NDIM,FACTOR)
*
* if ISCA(I).NE.0 VECUT(ISCA(I)) = VECOUT(ISCA(I))+VECIN(I)*SSCA(I)*FACT
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      DIMENSION VECIN(*),SSCA(*),ISCA(*)
*.Input and output
      DIMENSION VECOUT(*)
*
      DO 100 I = 1, NDIM
        IF(ISCA(I).NE.0) THEN
          VECOUT(ISCA(I)) =
     &    VECOUT(ISCA(I)) + VECIN(I)*SSCA(I)*FACTOR
        END IF
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE SHTOOB(NSHPIR,NIRREP,MXPOBS,NSMOB,NOSPIR,IOSPIR,
     &           NOBPS,NOB)
*
* Number of shells per irrep => Number of orbitals per symmetry
*
* =====
* Input
* =====
*
*  NSHPIR : Number of shells per irrep
*  NIRREP : Number of irreps
*  MXPOBS : Largest allowed number of orbitals symmetries
*  NSMOB  : Number of orbital symmetries
*  NOSPIR : Number of orbital symmetries per irrep
*  IOSPIR : Orbital symmetries per irrep
*
* ======
* Output
* ======
*  NOBPS  : Number of orbitals per symmetry
*  NOB    : Number of orbitals
*
* Jeppe Olsen, Winter of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION NSHPIR(*),NOSPIR(*),IOSPIR(MXPOBS,*)
*. Output
      DIMENSION NOBPS(*)
      CALL ISETVC(NOBPS,0,NSMOB)
      NOB = 0
      DO 100 IRREP = 1, NIRREP
        DO 90 ISM = 1, NOSPIR(IRREP)
          IISM = IOSPIR(ISM,IRREP)
          NOBPS(IISM) = NOBPS(IISM) + NSHPIR(IRREP)
          NOB = NOB + NSHPIR(IRREP)
   90   CONTINUE
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
         WRITE(6,*) ' SHTOOB Speaking '
         WRITE(6,*) ' =============== '
         WRITE(6,*) ' Number of orbitals obtained ', NOB
         WRITE(6,*) ' Number of orbitals per symmetry '
         CALL IWRTMA(NOBPS,1,NSMOB,1,NSMOB)
      END IF
*
      RETURN
      END
      SUBROUTINE SIGVST(ISGVST,NSMST)
*
* Obtain ISGVST(ISM): Symmetry of sigma v on string of symmetry ism
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER ISGVST(*)
*
      DO 100 ISM = 1, NSMST
C            MLSM(IML,IPARI,ISM,TYPE,IWAY)
        CALL MLSM(IML,IPARI,ISM,'ST',2)
        MIML = - IML
        CALL MLSM(MIML,IPARI,MISM,'ST',1)
        ISGVST(ISM) = MISM
  100 CONTINUE
*
      NTEST = 1
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' ISGVST array '
        WRITE(6,*) ' ============ '
        CALL IWRTMA(ISGVST,1,NSMST,1,NSMST)
      END IF
*
      RETURN
      END
      SUBROUTINE SLASK
      NAMELIST /LUCIA/ LUCTST
      READ(*,LUCIA)
      END
      SUBROUTINE SMOST(NSMST,NSMCI,MXPCSM,ISMOST)
*
* ISMOST(ISYM,ITOTSM): Symmetry of an internal state if ITOTSM
*                       if symmetry of 1 string is ISYM, the
*                       symmetry of the other string is
*                       ISMOST(ISYM,ITOTSM)
*
* Jeppe Olsen , Spring of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION ISMOST(MXPCSM,MXPCSM)
*
      DO 1000 ITOTSM = 1, NSMCI
       DO 900 ISTSM  = 1, NSMST
C            SYMCOM(ITASK,IOBJ,I1,I2,I12)
        CALL SYMCOM(2,1,ISTSM,JSTSM,ITOTSM)
        ISMOST(ISTSM,ITOTSM) = JSTSM
  900  CONTINUE
 1000 CONTINUE
*
      NTEST = 0
      IF( NTEST.NE. 0 ) THEN
        WRITE(6,*) ' ==============='
        WRITE(6,*) ' Info from SMOST '
        WRITE(6,*) ' ==============='
        DO 1010 ITOTSM = 1, NSMCI
          WRITE(6,*) ' ISMOST array for ITOTSM = ', ITOTSM
          CALL IWRTMA(ISMOST(1,ITOTSM),1,NSMST,1,NSMST)
 1010   CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE SMOSTB(NSMST,NSMCI,MXPCSM,ISMOST)
*
* ISMOST(ISYM,ITOTSM): Symmetry of an internal state is ITOTSM
*                       if symmetry of one string is ISYM, the
*                       symmetry of the other string is
*                       ISMOST(ISYM,ITOTSM)
*
* Jeppe Olsen , Spring of 1991
*
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION ISMOST(MXPCSM,MXPCSM)
*
      DO 1000 ITOTSM = 1, NSMCI
       DO 900 ISTSM  = 1, NSMST
C            SYMCOM(ITASK,IOBJ,I1,I2,I12)
        CALL SYMCOM(2,1,ISTSM,JSTSM,ITOTSM)
        ISMOST(ISTSM,ITOTSM) = JSTSM
  900  CONTINUE
 1000 CONTINUE
*
      NTEST = 0
      IF( NTEST.NE. 0 ) THEN
        DO 1010 ITOTSM = 1, NSMCI
          WRITE(6,*) ' ISMOST array for ITOTSM = ', ITOTSM
          CALL IWRTMA(ISMOST(1,ITOTSM),1,NSMST,1,NSMST)
 1010   CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE SORLOW(WRK,STVAL,ISTART,KZVAR,KEXST2,JEXSTV,IPRT)
C
C PURPOSE: FIND THE KEXSTV LOWEST VALUES IN WRK(KZVAR)
C          CHECK IF THE LAST ELEMENTS ARE DEGENERATE
C          JEXSTV IS THE NUMBER OF SORTED ELEMENTS WHERE
C          NO DEGENERACIES OCCUR AMONG THE HIGHEST ELEMENTS
* INPUT
*======
* WRK: ARRAY TO BE SORTED
* KEXST2: NUMBER OF ELEMENTS TO BE OBTAINED  + 1
* KZVAR: LENGTH OG WRK
* OUTPUT
*=======
* STVAL : STVAL(I) IS VALUES OF SORTED ELEMENT I
* ISTART: SCATTER POINTER ISTART(I) IS ADRESS IN FULL LIST OF
*          SORTED ELEMENT I
* KZVAR: LENGTH OG WRK
* JEXSTV: NUMBER OF ELEMENTS RETURNED , CAN BE LESS THAN
*          KEXSTV-1 IF DEGENERNCIES OCCURS  AMONG THE LAST ELEMENTS
*
C
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION WRK(*),STVAL(*),ISTART(*)
C
      PARAMETER ( TOLEQL=1.0D-6 )
      PARAMETER ( D0=0.0D0 , D1=1.0D0 , DM1 = -1.0D0 )
      LOGICAL FULL
C
      IF(KEXST2.GE.KZVAR) THEN
         FULL = .TRUE.
         KEXSTV = KZVAR
      ELSE
         FULL = .FALSE.
         KEXSTV = KEXST2+1
      END IF
C     write(6,*) ' SORLOW: KZVAR KEXST2 KEXSTV ',
C    &                      KZVAR,KEXST2,KEXSTV
*
      DO 100 K=1,KEXSTV
         ISTART(K) = K
         STVAL(K)  = WRK(K)
 100  CONTINUE
      KK=KEXSTV
C
      DO 210 I=1,KEXSTV
         DO 220 J=I+1,KEXSTV
            IF ((STVAL(J)-STVAL(I)) .LT. D0) THEN
               X =STVAL(I)
               II=ISTART(I)
               STVAL(I) =STVAL(J)
               ISTART(I)=ISTART(J)
               STVAL(J) =X
               ISTART(J)=II
            ENDIF
 220     CONTINUE
 210  CONTINUE
      GO TO 115
C     REPEAT UNTIL ...
 105  CONTINUE
         DO 110 I = KEXSTV,2,-1
            J = I - 1
            IF ((STVAL(J)-STVAL(I)) .GT. D0) THEN
               X =STVAL(I)
               II=ISTART(I)
               STVAL(I) =STVAL(J)
               ISTART(I)=ISTART(J)
               STVAL(J) =X
               ISTART(J)=II
            ELSE
               GO TO 115
            ENDIF
 110     CONTINUE
 115     CONTINUE
         STMAX=STVAL(KEXSTV)
C
 125     CONTINUE
            KK=KK+1
            IF (KK.LE.KZVAR) THEN
               IF ((STMAX-WRK(KK)).GT.D0) THEN
                  ISTART(KEXSTV) = KK
                  STVAL(KEXSTV)  = WRK(KK)
                  GO TO 105
C     ^--------------------
               ENDIF
               GO TO 125
C        ^--------------
            END IF
C
C     Check for degeneracy among diagonal elements
C
      I_DO_DEG_CHECK = 0
      JEXSTV = KEXSTV
      IF(I_DO_DEG_CHECK.EQ.1) THEN
      IF(.NOT.FULL) THEN
 
  160   JEXSTV = JEXSTV - 1
        IF (
     *  (ABS(STVAL(JEXSTV+1)-STVAL(JEXSTV))).LE.TOLEQL) GO TO 160
      END IF
      END IF! degeneracies should be checked
*
      IF((IPRT.GT.8).AND.(KEXST2.NE.JEXSTV)) WRITE(6,1600)KEXST2,JEXSTV
 
1600  FORMAT(/' NUMBER OF START VECTORS IS DIMINISHED TO',I5,' FROM',I5)
      IF (IPRT.GE.1000) THEN
         WRITE(6,*) '(I,(ISTART(I),STVAL(I)),I=1,JEXSTV)'
         DO 170 I = 1,JEXSTV
            WRITE(6,*) I,ISTART(I),STVAL(I)
  170    CONTINUE
         WRITE(6,*) 'THE FIRST',JEXSTV,' ELEMENTS ARE SELECTED.'
      END IF
C
C     END OF SORVAL
C
      RETURN
      END
      SUBROUTINE STSTSM(STSTSX,STSTDX,NSMST)
*
* construct  STSTSX and STSTDX giving
* symmetry of sx (dx) connecting two given string symmetries
*
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER STSTSX(NSMST,NSMST),STSTDX(NSMST,NSMST)
*
      DO 100 ILSTSM = 1, NSMST
        DO 50 IRSTSM = 1, NSMST
          CALL SYMCOM(1,5,ISXSM,IRSTSM,ILSTSM)
          CALL SYMCOM(1,6,IDXSM,IRSTSM,ILSTSM)
          STSTSX(ILSTSM,IRSTSM) = ISXSM
          STSTDX(ILSTSM,IRSTSM) = IDXSM
   50   CONTINUE
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' STSTSM: STSTSX, STSTDX '
        CALL IWRTMA(STSTSX,NSMST,NSMST,NSMST,NSMST)
        CALL IWRTMA(STSTDX,NSMST,NSMST,NSMST,NSMST)
      END IF
*
      RETURN
      END
      SUBROUTINE SXTYP(NSXTP,ITP,JTP,LEL1,LEL3,REL1,REL3)
*
* Types of creation and annihilation  operators so
* <L!a+ a!R> is nonvanishing
*
* L is defined by LEL1,LEL3
* R is defined by REL1,REL3
*
      INTEGER REL1,REL3
      INTEGER ITP(3),JTP(3)
      NSXTP = 0
*
*. To get rid of annoying and incorrect compiler warnings
      I1 = 0
      I3 = 0
      IJ1 = 0
      IJ3 = 0
*
      DO 100 I123 = 1, 3
        IF(I123.EQ.1) THEN
          I1 = 1
          I3 = 0
        ELSE IF(I123.EQ.2) THEN
          I1 = 0
          I3 = 0
        ELSE IF(I123.EQ.3) THEN
          I1 = 0
          I3 = 1
        END IF
        IF(LEL1-I1.LT.0) GOTO 100
        IF(LEL3-I3.LT.0) GOTO 100
        DO 50 J123 = 1, 3
          IF(J123.EQ.1) THEN
            IJ1 = I1 - 1
            IJ3 = I3
          ELSE IF(J123.EQ.2) THEN
            IJ1 = I1
            IJ3 = I3
          ELSE IF(J123.EQ.3) THEN
            IJ1 = I1
            IJ3 = I3-1
          END IF
          IF(REL1+IJ1.EQ.LEL1.AND.REL3+IJ3.EQ.LEL3) THEN
            NSXTP = NSXTP + 1
            ITP(NSXTP) = I123
            JTP(NSXTP) = J123
          END IF
   50   CONTINUE
  100 CONTINUE
*
      NTEST = 0
      IF(NTEST.NE.0) THEN
        WRITE(6,'(A,4I4)')
     &  ' SX  connecting LEL1,LEL3,REL1,REL3 ',LEL1,LEL3,REL1,REL3
        WRITE(6,*) ' Number of connections obtained ', NSXTP
        WRITE(6,*) ' ITYPE JTYPE '
        WRITE(6,*) ' =========== '
        DO 200 I = 1, NSXTP
         WRITE(6,'(2I5)') ITP(I),JTP(I)
  200   CONTINUE
*
      END IF
*
      RETURN
      END
      SUBROUTINE SYMCM1(ITASK,IOBJ,I1,I2,I12)
*
* Symmetries I1,I2,I12 are related as
* I1*I2 = 12
* IF(ITASK = 1 ) I2 and I12 are known, find I1
* IF(ITASK = 2 ) I1 and I12 are known, find I1
* IF(ITASK = 3 ) I1 and I2 are known , find I12
*
* D2h version , written for compatibility with general symmetry
*
      INTEGER SYMPRO(8,8)
      DATA  SYMPRO/1,2,3,4,5,6,7,8,
     &             2,1,4,3,6,5,8,7,
     &             3,4,1,2,7,8,5,6,
     &             4,3,2,1,8,7,6,5,
     &             5,6,7,8,1,2,3,4,
     &             6,5,8,7,2,1,4,3,
     &             7,8,5,6,3,4,1,2,
     &             8,7,6,5,4,3,2,1 /
*
      IF(ITASK.EQ.1) THEN
        I1 = SYMPRO(I2,I12)
*
C?    IF(I12.GT.8.OR.I2.GT.8.OR.I12.LE.0.OR.I2.LE.0) THEN 
C?      WRITE(6,*) ' I12 and I2 = ', I12, I2
C?    END IF
*
      ELSE IF(ITASK.EQ.2) THEN
*
C?    IF(I12.GT.8.OR.I1.GT.8.OR.I12.LE.0.OR.I1.LE.0) THEN 
C?      WRITE(6,*) ' I12 and I1 = ', I12, I1
C?    END IF
*
        I2 = SYMPRO(I1,I12)
      ELSE IF (ITASK.EQ.3) THEN
*
C?    IF(I2.GT.8.OR.I1.GT.8.OR.I2.LE.0.OR.I1.LE.0) THEN 
C?      WRITE(6,*) ' I2 and I1 = ', I2, I1
C?    END IF
*
        I12 = SYMPRO(I1,I2)
*
      END IF
*
      RETURN
      END
      SUBROUTINE SYMCOM(ITASK,IOBJ,I1,I2,I12)
*
* Symmetries I1,I2,I12 are related as
* I1 I2 = 12
* IF(ITASK = 1 ) I2 and I12 are known, find I1
* IF(ITASK = 2 ) I1 and I12 are known, find I1
* IF(ITASK = 3 ) I1 and I2 are known , find I12
*
* IOBJ = 1: I1,I2 are strings I12 determinant
* ( Other things can follow )
* IOBJ = 2: I1,I2,I3 are externals
* IOBJ = 3: I1 is an external, I2,I3 are dets
* IOBJ = 4: I1 is orbital, I2 is string,l, I12 is string
* IOBJ = 5: I1 is single excitation, I2 is string,l, I12 is string
* IOBJ = 6: I1 is orbital, I2 is Orbital I12 is single excitation 
*
* If obtained symmetry I1 or I2 is outside bounds,
* zero is returned.
*
* Jeppe Olsen , Spring of 1991
*
* ================
*. Driver routine
* ================
*.LUCINP (PNTGRP is used )
C     INTEGER PNTGRP,EXTSPC
C     PARAMETER(MXPIRR = 20)
C     PARAMETER ( MXPOBS = 20 )
C     PARAMETER (MXPR4T = 10 )
C     COMMON/LUCINP/PNTGRP,NIRREP,NSMCMP,MAXML,MAXL,
C    &              INTSPC,EXTSPC,NRSSH(MXPIRR,3),
C    &              MNRS1R,MXRS1R,MNRS3R,MXRS3R,NACTEL,
C    &              NSMOB,NRS0SH(1,MXPIRR),NRS4SH(MXPR4T,MXPIRR),
C    &              MXR4TP, MXHR0,MXER4,
C    &              NINASH(MXPIRR),
C    &              INTXCI,NDELSH(MXPIRR),MNRS10,MXRS30
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*
      IF(PNTGRP.EQ.1) THEN
        CALL SYMCM1(ITASK,IOBJ,I1,I2,I12)
      ELSE IF(PNTGRP.GE.2.AND.PNTGRP.LE.4) THEN
        CALL SYMCM2(ITASK,IOBJ,I1,I2,I12)
      ELSE
        WRITE(6,*) ' PNTGRP parameter out of bounds ', PNTGRP
        WRITE(6,*) ' Enforced stop in SYMCOM '
        STOP 11
      END IF
*
      RETURN
      END
      SUBROUTINE SYMINF(IPRNT)
*
* Information about number of symmetries
*
* Input: /LUCINP/,/ORBINP
* Output: /CSM/,/CSMPRO/
*
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
C     PARAMETER(MXPIRR = 20)
C     PARAMETER ( MXPOBS = 20 )
C     PARAMETER (MXPR4T = 10 )
C     INTEGER PNTGRP,EXTSPC
C     COMMON/LUCINP/PNTGRP,NIRREP,NSMCMP,MAXML,MAXL,
C    &              INTSPC,EXTSPC,NRSSH(MXPIRR,3),
C    &              MNRS1R,MXRS1R,MNRS3R,MXRS3R,NACTEL,
C    &              NSMOB,NRS0SH(1,MXPIRR),NRS4SH(MXPR4T,MXPIRR),
C    &              MXR4TP, MXHR0,MXER4,
C    &              NINASH(MXPIRR),
C    &              INTXCI,NDELSH(MXPIRR),MNRS10,MXRS30
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*
 
*. Output
* NSMSX: number of symmetries of single excitations
* NSMDX: Number of symmetries of double excitations
* NSMST : Number of symmetries of strings
* NSMCI : NUmber of symmetries of CI spaces
* ITSSX : Total symmetrix single excitation
* ITSDX : Total symmetrix double excitation
C     COMMON/CSM/NSMSX,NSMDX,NSMST,NSMCI,ITSSX,ITSDX
      INCLUDE 'csm.inc'
*
      INCLUDE 'csmprd.inc'
c      INTEGER ADASX,ASXAD,ADSXA,SXSXDX,SXDXSX
c      COMMON/CSMPRD/ADASX(MXPOBS,MXPOBS),ASXAD(MXPOBS,2*MXPOBS),
c     &              ADSXA(MXPOBS,2*MXPOBS),
c     &              SXSXDX(2*MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
* ADASX : symmetry of orbs i and i => symmetry of a+iaj
* ASXAD : symmetry of orb j and excit a+iaj => symmetry of i
* ADSXA : symmetry of orb i and excit a+iaj => symmetry of j
*
* SXSXDX : Symmetry of two single excitations
*          => symmetry of double  excitation
* SXDXSX : Symmetry of single excitation and double excitation
*          => symmetry of single  excitation
 
*.
      IF(PNTGRP.EQ.1) THEN
* =====
* D 2 h
* =====
        CALL ZSYM1(NIRREP,IPRNT)
      ELSE IF(PNTGRP.EQ.2) THEN
* ========
* C inf V
* ========
C            ZNONAB(INVCEN,MAXMLO,NSMOB)
        CALL ZNONAB(0,MAXML,NSMOB,IPRNT)
        CALL ZSYM2(IPRNT)
      ELSE IF(PNTGRP.EQ.3.OR.PNTGRP.EQ.4) THEN
* ===========
* D inf H O3
* ===========
C            ZNONAB(INVCEN,MAXMLO,NSMOB)
        CALL ZNONAB(1,MAXML,NSMOB,IPRNT)
        CALL ZSYM2(IPRNT)
      ELSE
        WRITE(6,*) ' You are to early , sorry '
        WRITE(6,*) ' Illegal PNTGRP in SYMINF ',PNTGRP
        STOP 11
      END IF
*
      RETURN
      END
      SUBROUTINE TRIPK2(AUTPAK,APAK,IWAY,MATDIM,NDIM,SIGN)
C
C
C.. REFORMATING BETWEEN LOWER TRIANGULAR PACKING
C   AND FULL MATRIX FORM FOR A SYMMETRIC OR ANTI SYMMETRIC MATRIX
C
C   IWAY = 1: FULL TO PACKED
C              LOWER HALF OF AUTPAK IS STORED IN APAK
C   IWAY = 2: PACKED TO FULL FORM
C              APAK STORED IN LOWER HALF
C               SIGN * APAK TRANSPOSED IS STORED IN UPPPER PART
C.. NOTE: COLUMN WISE STORAGE SCHEME IS USED FOR PACKED BLOCKS
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION AUTPAK(MATDIM,MATDIM),APAK(*)
C
      IF( IWAY .EQ. 1 ) THEN
        IJ = 0
        DO 100 J = 1,NDIM
          DO 50  I = J , NDIM
           APAK(IJ+I) = AUTPAK(I,J)
   50     CONTINUE
          IJ = IJ +NDIM-J
  100   CONTINUE
      END IF
C
      IF( IWAY .EQ. 2 ) THEN
        IJ = 0
        DO 200 J = 1,NDIM
          DO 150  I = J,NDIM
           AUTPAK(J,I) = SIGN*APAK(IJ+I)
           AUTPAK(I,J) = APAK(IJ+I)
  150     CONTINUE
          IJ = IJ + NDIM-J
  200   CONTINUE
      END IF
C
      NTEST = 0
      IF( NTEST .NE. 0 ) THEN
        WRITE(6,*) ' AUTPAK AND APAK FROM TRIPK2 '
        CALL WRTMAT(AUTPAK,NDIM,MATDIM,NDIM,MATDIM)
        CALL PRSM2(APAK,NDIM)
      END IF
C
      RETURN
      END
      SUBROUTINE UPPCAS(LINE,LENGTH)
*
* Convert letters in character string LINE to upper case
*
* very stupid and not vectorized !
*
      CHARACTER*(*) LINE
      PARAMETER (NCHAR = 43)
      CHARACTER*1 LOWER(NCHAR)
      CHARACTER*1 UPPER(NCHAR)
*
      DATA LOWER/'a','b','c','d','e',
     &           'f','g','h','i','j',
     &           'k','l','m','n','o',
     &           'p','q','r','s','t',
     &           'u','v','w','x','y',
     &           'z','+','-','<','>',
     &           '=','0','1','2','3',
     &           '4','5','6','7','8',
     &           '9','_',' '/
      DATA UPPER/'A','B','C','D','E',
     &           'F','G','H','I','J',
     &           'K','L','M','N','O',
     &           'P','Q','R','S','T',
     &           'U','V','W','X','Y',
     &           'Z','+','-','<','>',
     &           '=','0','1','2','3',
     &           '4','5','6','7','8',
     &           '9','_',' '/
*
      DO 100 ICHA = 1, LENGTH
        DO 50 I = 1,NCHAR
          IF(LINE(ICHA:ICHA).EQ.LOWER(I))
     &    LINE(ICHA:ICHA) = UPPER(I)
   50   CONTINUE
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE WEIGHT(Z,NEL,NORB1,NORB2,NORB3,
     &                  MNRS1,MXRS1,MNRS3,MXRS3,ISCR,NTEST)
*
* construct vertex weights
*
* Reverse lexical ordering is used for restricted space
*
      IMPLICIT REAL*8           ( A-H,O-Z)
      INTEGER Z(*), ISCR(*)
*
      NORB = NORB1 + NORB2 + NORB3
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' >>>> WEIGHT <<<<< '
        WRITE(6,*) ' NORB1 NORB2 NORB3 ',NORB1,NORB2,NORB3
        WRITE(6,*) ' NEL MNRS1 MXRS1 MNRS3 MXRS3 '
        WRITE(6,*)   NEL,MNRS1,MXRS1,MNRS3,MXRS3
      END IF
*
      KLFREE = 1
      KLMAX = KLFREE
      KLFREE = KLFREE + NORB
*
      KLMIN = KLFREE
      KLFREE = KLFREE + NORB
*
      KW = KLFREE
      KLFREE = KW + (NEL+1)*(NORB+1)
*.Max and min arrays for strings
      CALL RSMXMN(ISCR(KLMAX),ISCR(KLMIN),NORB1,NORB2,NORB3,
     &            NEL,MNRS1,MXRS1,MNRS3,MXRS3,NTEST)
*. Arc weights
      CALL GRAPW(ISCR(KW),Z,ISCR(KLMIN),ISCR(KLMAX),NORB,NEL,NTEST)
*
      RETURN
      END
C                WRSVCD(LU1,LBLK,VEC1,ISCR1(IBASE),SCR1,NDEG,NDIM,
C    &           LUDIA)
      SUBROUTINE WRSVCD(LU,LBLK,VEC1,IPLAC,VAL,NSCAT,NDIM,LUFORM,JPACK)
*
* Write scattered vector to disc.
*.Vector is always written in packed form
*
      IMPLICIT REAL*8(A-H,O-Z)
*.Input
      DIMENSION IPLAC(*),VAL(*)
*.Scratch
      DIMENSION VEC1(*)
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' Info from WRSVCD'
        WRITE(6,*) ' ================'
        WRITE(6,*)
        WRITE(6,*) ' Number of elements: ', NSCAT  
        WRITE(6,*)
      END IF
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Addresses of elements:'
        CALL IWRTMA(IPLAC,1,NSCAT,1,NSCAT)
        WRITE(6,*) ' Values of elements:'
        CALL WRTMAT(VAL,1,NSCAT,1,NCAT)
      END IF
*
      IF(LBLK.GT.0) THEN
*. Write the vector without markers in one block
         CALL SETVEC(VEC1,0.0D0,NDIM)
         DO 100 IEFF = 1, NSCAT
           VEC1(IPLAC(IEFF)) = VAL(IEFF)
  100    CONTINUE
         CALL TODSC(VEC1,NDIM,-1,LU)
      ELSE
*. Write the vector with the block format of file LUFORM
        CALL REWINO(LUFORM)
        IBOT = 1
*. Loop over records
 1000   CONTINUE
*.Length
        CALL IFRMDS(LBL,1,-1,LUFORM)
        CALL ITODS (LBL,1,-1,LU)
        IF(LBL.GE.0) THEN
C?        write(6,*) ' IBOT, LBL  ',IBOT, LBL  
          CALL SETVEC(VEC1,0.0D0,LBL)
          DO 200 IEFF = 1, NSCAT
            IF( IPLAC(IEFF).GE.IBOT.AND.IPLAC(IEFF).LE.IBOT+LBL-1) 
     &      VEC1(IPLAC(IEFF)-IBOT+1) = VAL(IEFF)
C?          IF( IPLAC(IEFF).GE.IBOT.AND.IPLAC(IEFF).LE.IBOT+LBL-1) 
C?   &      write(6,*) ' Catch: IPLAC(IEFF) VAL(IEFF) ',
C?   &      IPLAC(IEFF),VAL(IEFF)
  200     CONTINUE
C         write(6,*)  'record to be disced '
C         CALL WRTMAT(VEC1,1,LBL,1,LBL)
          IF(JPACK.EQ.1) THEN
            CALL TODSCP(VEC1,LBL,-1,LU)
          ELSE 
            CALL TODSC(VEC1,LBL,-1,LU)
          END IF
*.Skip record on LUFORM
          CALL FRMDSC(VEC1,LBL,-1,LUFORM,IMZERO,IAMPACK)
          IBOT = IBOT + LBL
          GOTO 1000
        END IF
*
C       WRITE(6,*) ' output from WRSVCD '
C       CALL WRTVCD(VEC1,LU,1,LBLK)
      END IF
*
      RETURN
      END
      SUBROUTINE WRTHD(LUOUT)
CJO-Start
*
* Introduce LUCIA
*
      CALL PRINTVERSION()
      WRITE(6,'(1H ,8X,A)')
     &'****************************************************************'
      WRITE(6,'(1H ,8X,A)')
     &'*                                                              *'
      WRITE(6,'(1H ,8X,A)')
     &'*           Welcome to LUCIA                                   *'
      WRITE(6,'(1H ,8X,A)')
     &'*                                                              *'
      WRITE(6,'(1H ,8X,A)')
     &'* Written by Jeppe Olsen, University of Aarhus, Denmark        *'
      WRITE(6,'(1H ,8X,A)')
     &'* Contributing authors:                                        *'
      WRITE(6,'(1H ,8X,A)')
     &'*          Andreas Koehn, University of Aarhus, Denmark        *'
      WRITE(6,'(1H ,8X,A)')
     &'* Version of May. 3, 2013  (see also compilation mark above)   *'
      WRITE(6,'(1H ,8X,A)')
     &'****************************************************************'
*
      WRITE(6,*)
      WRITE(6,'(1H ,8X,A)')
     &' In case of trouble please contact: '
      WRITE(6,'(1H ,12X,2A)')
     &' Jeppe Olsen, Dept. of Chemistry, University of Aarhus,',
     &' Aarhus, DK-8000 Denmark ' 
      WRITE(6,'(1H ,12X,A)')
     &' Telephone: +45 23382435'
      WRITE(6,'(1H ,12X,A)')
     &' E-mail: jeppe@chem.au.dk   '              
      WRITE(6,*)
      RETURN
      END
      SUBROUTINE WRTRS2(VECTOR,ISMOST,ICBLTP,IOCOC,NOCTPA,NOCTPB,
     &                  NSASO,NSBSO,NSMST)
*
* Write RAS vector . Storage form is defined by ICBLTP
*
      IMPLICIT REAL*8           (A-H,O-Z)
*
      DIMENSION VECTOR(*)
      DIMENSION IOCOC(NOCTPA,NOCTPB)
      DIMENSION NSASO(NSMST,* ),NSBSO(NSMST,* )
      DIMENSION ICBLTP(*),ISMOST(*)
*
*
      IBASE = 1
      DO 1000 IASM = 1, NSMST
        IBSM = ISMOST(IASM)
        IF(IBSM.EQ.0.OR.ICBLTP(IASM).EQ.0) GOTO 1000
*
        DO 900 IATP = 1, NOCTPA
          IF(ICBLTP(IASM).EQ.2) THEN
            IBTPMX = IATP
          ELSE
            IBTPMX = NOCTPB
          END IF
          NAST = NSASO(IASM,IATP)
*
          DO 800 IBTP = 1 , IBTPMX
            IF(IOCOC(IATP,IBTP) .EQ. 0 ) GOTO 800
            NBST = NSBSO(IBSM,IBTP)
            IF(ICBLTP(IASM).EQ.2.AND.IATP.EQ.IBTP ) THEN
* Diagonal block
              NELMNT = NAST*(NAST+1)/2
              IF(NELMNT.NE.0) THEN
                WRITE(6,'(A,3I3)')
     &          '  Iasm iatp ibtp: ', IASM,IATP,IBTP
                WRITE(6,'(A)')
     &          '  ============================'
                CALL PRSM2(VECTOR(IBASE),NAST)
                IBASE = IBASE + NELMNT
              END IF
            ELSE
              NELMNT = NAST*NBST
              IF(NELMNT.NE.0) THEN
                WRITE(6,'(A,3I3)')
     &          '  Iasm iatp ibtp: ', IASM,IATP,IBTP
                WRITE(6,'(A)')
     &          '  ============================'
                CALL WRTMAT(VECTOR(IBASE),NAST,NBST,NAST,NBST)
                IBASE = IBASE + NELMNT
              END IF
            END IF
  800     CONTINUE
  900   CONTINUE
 1000 CONTINUE
*
      RETURN
      END
      SUBROUTINE XDIXT2(XDX,X,DIA,NXRDM,NXCDM,SHIFT,SCR)
*
* Obtain XDX = X * (DIA+Shift)-1 * X(Transposed)
* where DIA is an diagonal matrix stored as a vector
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION XDX(NXRDM,NXRDM)
      DIMENSION X(NXRDM,NXCDM),DIA(NXCDM)
      DIMENSION SCR(NXCDM)
*
      NTEST = 000
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' Info from XDIXT2 '
        WRITE(6,*) ' ================='
        WRITE(6,*)
        WRITE(6,*) ' NXRDM, NXCDM, SHIFT = ',
     &               NXRDM, NXCDM, SHIFT
      END IF
*
      CALL SETVEC(XDX,0.0D0,NXRDM ** 2 )
      THRES = 1.0D-9
      DO 100 J=1,NXRDM
*.Scr(k) = X(J,K)/(SHIFT+DIA(K)
        DO 10 K = 1, NXCDM
          IF(ABS(SHIFT+DIA(K)) .GT. THRES ) THEN
            SCR(K) = X(J,K)/(SHIFT+DIA(K))
          ELSE
            SCR(K) = X(J,K)/THRES
          END IF
   10   CONTINUE
*
        DO 20 K = 1, NXCDM
          FACTOR = SCR(K)
          CALL VECSUM(XDX(1,J),XDX(1,J),X(1,K),1.0D0,FACTOR,NXRDM)
   20   CONTINUE
*
  100 CONTINUE
*
      NTEST = 00
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' x (Dia + shift ) - 1 X(T) from XDIXT2 '
        CALL WRTMAT(XDX,NXRDM,NXRDM,NXRDM,NXRDM)
      END IF
*
      RETURN
      END
      SUBROUTINE XDXTV(VECUT,VECIN,X,DIA,NDIM,SCR,SHIFT,IINV)
*
* VECUT = X (DIA+SHIFT)* XT* VECIN FOR IINV = 0
* VECUT = X (DIA+SHIFT)**-1 *XT *VECIN FOR IINV = 1
* where DIA is an diagonal matrix stored as a vector
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION VECUT(*),VECIN(*)
      DIMENSION X(NDIM,NDIM),DIA(NDIM)
      DIMENSION SCR(*)
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
       WRITE(6,*) ' INFO FROM XDXTV '
       WRITE(6,*) '================='
       WRITE(6,*)
       WRITE(6,*)'X AND DIA '
       CALL WRTMAT(X,NDIM,NDIM,NDIM,NDIM)
       CALL WRTMAT(DIA,1,NDIM,1,NDIM)
      END IF
*
*. SCRATCH MUST BE ABLE TO HOLD A VECTOR OF LENGTH NDIM
*. X(T)*VECIN
      CALL MATVCB(X,VECIN,VECUT,NDIM,NDIM,1)
* DIAGONAL-TERM * XT*VECIN
      IF(IINV.EQ.0) THEN
        CALL VVTOV(VECUT,DIA,SCR,NDIM)
        CALL VECSUM(SCR,VECUT,SCR,SHIFT,1.0D0,NDIM)
      ELSE
        CALL DIAVC3(SCR,VECUT,DIA,SHIFT,NDIM,VDSV)
      END IF
*. MULTIPLY WITH X
      CALL MATVCB(X,SCR,VECUT,NDIM,NDIM,0)
C     DIAVC3(VECOUT,VECIN,DIAG,SHIFT,NDIM,VDSV)
C     MATVCB(MATRIX,VECIN,VECOUT,MATDIM,NDIM,ITRNSP)
C     VVTOV(VECIN1,VECIN2,VECUT,NDIM)
*
      RETURN
      END
      SUBROUTINE ZBASE(NVEC,IVEC,NCLASS)
*
*  Some class division exists with NVEC(ICLASS) members in
*  class ICLASS.
*
*  Construct array IVEC(ICLASS) giving first element of
*  class ICLASS in full adressing
*
      IMPLICIT REAL*8          (A-H,O-Z)
      DIMENSION NVEC(*),IVEC(*)
*
      IVEC(1) = 1
      DO 100 ICLASS = 2,NCLASS
        IVEC(ICLASS) = IVEC(ICLASS-1)+NVEC(ICLASS-1)
  100 CONTINUE
*
      NTEST = 0
      IF( NTEST .NE. 0 ) THEN
        WRITE(6,'(A)') '  ZBASE: NVEC and IVEC '
        WRITE(6,'(A)') '  ===================== '
        CALL IWRTMA(NVEC,1,NCLASS,1,NCLASS)
        CALL IWRTMA(IVEC,1,NCLASS,1,NCLASS)
      END IF
*
      RETURN
      END
*
* Codes for general symmetry handling
*
*                - ZSTINF : generate /STINF/ info on strings and mapping
*                - MEMSTR : allocates memory for string information
*                - WEIGHT : Weights for strings
*                - NSTRSO : Number of strings per sym and class
*                - ZBASE  : offset arrays for strings
*                - ZSMCL  : symmetry and class for each string
*                - GENSTR : Generate strings ordered by sym and class
*                - MEMEXT : Memory for external blocks
*
      SUBROUTINE ZBLTP(ISMOST,MAXSYM,IDC,ICBLTP,IMMLST)
*
* Generate vector ICBLTP giving type of each block
*
*
* ICBLTP gives type of symmetry block :
* = 0 : symmetry block is not included
* = 1 : symmetry block is included , all OO types
* = 2 : symmetry block is included , lower OO types
*
*. Input
      DIMENSION ISMOST(*),IMMLST(*)
*. Output
      DIMENSION ICBLTP(*)
*. Changed to simplify structure for IDC .le. 2
      IF(IDC.LE.2) THEN
*. No spatial degeneracy
        DO IASYM = 1, MAXSYM
          IBSYM = ISMOST(IASYM)
          IF(IDC.EQ.2.AND.IBSYM.GT.IASYM) THEN
*.Symmetry block excluded
            ICBLTP(IASYM) = 0
          ELSE IF((IDC.EQ.2.AND.IASYM.GT.IBSYM).OR.IDC.EQ.1) THEN
*.Complete symmetry block included
            ICBLTP(IASYM) = 1
          ELSE
*.Lower half  symmetry block included
            ICBLTP(IASYM) = 2
          END IF
        END DO
      ELSE
*. Also spatial degeneracy
      DO 100 IASYM = 1, MAXSYM
*
        IBSYM = ISMOST(IASYM)
        IF(IBSYM .EQ. 0 ) GOTO 100
        IF(((IDC.EQ.2.OR.IDC.EQ.4).AND.(IBSYM.GT.IASYM))
     &                    .OR.
     &       (IDC.EQ.3.AND.IMMLST(IASYM).GT.IASYM)) THEN
*.Symmetry block excluded
          ICBLTP(IASYM) = 0
        ELSE IF((IDC.EQ.2.AND.IASYM.GT.IBSYM)
     &                   .OR.
     &                IDC.EQ.1
     &                   .OR.
     &          (IDC.EQ.3.AND.IASYM.GE.IMMLST(IASYM))) THEN
*.Complete symmetry block included
          ICBLTP(IASYM) = 1
        ELSE
*.Lower half  symmetry block included
          ICBLTP(IASYM) = 2
        END IF
  100 CONTINUE
      END IF
*     ^ End of IDC switch 
*
      NTEST = 0
      IF ( NTEST .NE. 0 ) THEN
         WRITE(6,*)
         WRITE(6,*) ' Output from ZBLTP '
         WRITE(6,*) ' =================='
         WRITE(6,*)
         WRITE(6,*) ' IDC = ', IDC
         WRITE(6,*) ' ISMOST: '
         CALL IWRTMA(ISMOST,1,MAXSYM,1,MAXSYM)
         WRITE(6,*) ' Block type of symmetry blocks '
         CALL IWRTMA(ICBLTP,1,MAXSYM,1,MAXSYM)
      END IF
*
      RETURN
      END
      SUBROUTINE ZNONAB(INVCEN,MAXMLO,NSMOB,IPRNT)
*
*
* ============================
* Set up common block /NONAB/
* ============================
*
*========
* Input :
*========
*      INVCNT :inversion center is present(1), absent(0)
*      MAXMLO : Largest ML value of orbitals
*      NSMOB  : Number of symmetries of orbitals
*      Contents of common block /STRINP/,/ORBINP/
*=========
* output :
*=========
*======================
* Orbital Information
*======================
*      NORASM : Number of orbitals per abelian symmetry
*      MNMLOB : Smallest ML of orbitals
*      MXMLOB : largest ML of orbitals
*      NMLOB  : number of ML values for orbitals
*
*======================
* String Information
*======================
*      MNMLST : smallest ML of any strings
*      MXMLST : largest ML of any strings
*      NMLST  : Number of ML values of strings
*      NSMST  : Number of symmetries of strings
*
*==============================
* Single excitation Information
*==============================
*      MNMLSX : SMALLEST ML OF SINGLE EXCITATION
*      MXMLSX : LARGEST ML OF SINGLE EXCITATIONS
*      NMLSX  : NUMBER OF ML VALUES FOR SINGLE EXCITATIONS
*      NSMSX  : NUMBER OF SYMMETRIES FOR SINGLE EXCITATIONS
*=============================================
* External configurations (upto 4 electrons )
*=============================================
*      MNMLXT : SMALLEST ML OF External configurations
*      MXMLSX : LARGEST ML OF external configurations
*      NMLXT  : NUMBER OF ML VALUES FOR ext. configurations
*      NSMXT  : NUMBER OF SYMMETRIES FOR ext. configurations
*
* =============
* General input
* =============
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*./CSM/
C     COMMON/CSM/NSMSX,NSMDX,NSMST,NSMCI,ITSSX,ITSDX
      INCLUDE 'csm.inc'
*./STRINP/
C     PARAMETER(MXPSTT=40)
C     COMMON/STRINP/NSTTYP,MNRS1(MXPSTT),MXRS1(MXPSTT),
C    &              MNRS3(MXPSTT),MXRS3(MXPSTT),NELEC(MXPSTT),
C    &              IZORR(MXPSTT),IAZTP,IBZTP,IARTP(3,10),IBRTP(3,10),
C    &              NZSTTP,NRSTTP,ISTTP(MXPSTT)
      INCLUDE 'strinp.inc'
* =======
*. Output
* =======
*./NONAB/
      LOGICAL INVCNT
      COMMON/NONAB/ INVCNT,NIG,NORASM(MXPOBS),
     &              MNMLOB,MXMLOB,NMLOB,
     &              MXMLST,MNMLST,NMLST,
     &              NMLSX ,MNMLSX,MXMLSX,
     &              MNMLCI,MXMLCI,NMLCI,
     &              MXMLDX,MNMLDX,NMLDX
*
      NTEST = 0
      NTEST = MAX(IPRNT,NTEST)
*. Inversion symmetry
      IF( INVCEN .EQ. 0 ) THEN
        INVCNT = .FALSE.
        NIG = 1
      ELSE
        INVCNT = .TRUE.
        NIG = 2
      END IF
*
** 1 : Information about orbitals
*
      MXMLOB = MAXMLO
      MNMLOB =-MAXMLO
      NMLOB = MXMLOB - MNMLOB + 1
*. Number of orbitals per symmetry
      DO 10 ISYM = 1, NSMOB
        NORASM(ISYM) = IFREQ(ISMFTO,ISYM,NACOB)
   10 CONTINUE
      IF( NTEST.GE. 2 ) THEN
        WRITE(6,*) ' NORASM '
        CALL IWRTMA(NORASM,1,NSMOB,1,NSMOB)
        WRITE(6,*) ' MNMLOB,MXMLOB ',MNMLOB,MXMLOB
        WRITE(6,*) ' NMLOB, NSMOB ',NMLOB,NSMOB
      END IF
*
**  2. Information about strings
*
      MXMLST = 0
      MNMLST = 0
      DO 50 ITYPE = 1, NSTTYP
        IEL = NELEC(ITYPE)
*
        MXMLTP = 0
        DO 40 IML = MXMLOB,MNMLOB,-1
          IORB = NORASM(IML-MNMLOB+1)
          IF(INVCNT) IORB = IORB + NORASM(NMLOB+IML-MNMLOB+1)
          IEL2 = MIN(IORB,IEL)
          MXMLTP = MXMLTP + IEL2*IML
          IEL = IEL - IEL2
   40   CONTINUE
        MXMLST = MAX(MXMLST,MXMLTP)
*
        MNMLTP = 0
        IEL = NELEC(ITYPE)
        DO 45 IML = MNMLOB,MXMLOB
          IORB = NORASM(IML-MNMLOB+1)
          IF(INVCNT) IORB = IORB + NORASM(NMLOB+IML-MNMLOB+1)
          IEL2 = MIN(IORB,IEL)
          MNMLTP = MNMLTP + IEL2*IML
          IEL = IEL - IEL2
   45   CONTINUE
        MNMLST = MIN(MNMLST,MNMLTP)
   50 CONTINUE
*
      NMLST  = MXMLST - MNMLST + 1
      NSMST  = NIG * NMLST
*
      IF( NTEST .GE. 2 ) THEN
        WRITE(6,*) ' MXMLST,MNMLST,NSMST'
        WRITE(6,*)   MXMLST,MNMLST,NSMST
      END IF
*
** 3. Information about single excitations
*
      MNMLSX = MNMLOB - MXMLOB
      MXMLSX = MXMLOB - MNMLOB
      NMLSX  = MXMLSX - MNMLSX +1
      NSMSX  = NIG * NMLSX
 
      IF( NTEST .GE.2 ) THEN
        WRITE(6,*) ' NMLSX,NSMSX,MNMLSX ',NMLSX,NSMSX,MNMLSX
      END IF
*
** 4 : External configurations(double excitations)
*
      MXMLDX = 4*MAXMLO
      MNMLDX = (-4)*MAXMLO
      NMLDX  = MXMLDX - MNMLDX + 1
      NSMDX  = NIG * NMLDX
      IF( NTEST .GE.2 ) THEN
        WRITE(6,*) ' NMLDX,NSMDX,MNMLDX ',NMLDX,NSMDX,MNMLDX
      END IF
*
** 5 : Determinants
*
      MXMLCI =  2*MXMLST + MXMLDX
      MNMLCI = - MXMLCI
      NMLCI = 2 * MXMLCI + 1
      NSMCI = NIG * NMLCI
*
*.6 Total symmetrix single excitation and external
*
      ITSSX = 0 - MNMLSX + 1
      ITSDX = 0 - MNMLDX + 1
 
      IF ( NTEST .GE. 1 ) THEN
        WRITE(6,*)
        WRITE(6,'(A,I4)')
     &  '  Number of symmetries of orbitals     .. ', NSMOB
        WRITE(6,'(A,I4)')
     &  '  Number of symmetries of strings      .. ', NSMST
        WRITE(6,'(A,I4)')
     &  '  Number of symmetries of single excit. . ', NSMSX
        WRITE(6,'(A,I4)')
     &  '  Number of symmetries of double excit. . ', NSMDX
        WRITE(6,'(A,I4)')
     &  '  Number of symmetries of determinants .. ', NSMCI
        WRITE(6,*)
*
        WRITE(6,*) ' Total symmetric single excitation .. ',ITSSX
        WRITE(6,*) ' Total symmetric double excitation .. ',ITSDX
      END IF
*
      RETURN
      END
 
      SUBROUTINE ZOOS(ISMOST,IBLTP,MAXSYM,IOCOC,NSSOA,NSSOB,
     &                NOCTPA,NOCTPB,IDC,IOOS,NOOS,NCOMB,IXPND)
*
* Generate offsets for CI vector for RAS CI expansion of given symmetry
* Combination type is defined by IDC
* Total number of combinations NCOMB is also obtained
*
* Symmetry is defined through ISMOST
*
* ICBLTP gives typo of symmetry block:
* = 0: symmetry block is not included
* = 1: symmetry block is included , all OO types
* = 2: symmetry block is included , lower OO types
*
* If IXPND .ne. 0 , the diagonal blocks are always choosen expanded
*
* ========
*  Output
* ========
*
* IOOS(IOCA,IOCB,ISYM): Start of block with alpha strings of
*                        symmetry ISYM and type IOCA, and
*                        betastrings of type IOCB
* NOOS(IOCA,IOCB,ISYM): Number of combinations
* The ordering used for the CI vector is
*
*    SYMMETRY  FOR ALPHA STRINGS..(GIVES SYMMETRY OF BETA STRING )
*         OCCUPATION TYPE  FOR ALPHA STRING
*            OCCUPATION TYPE FOR    BETA STRING
*                BETA STRING ( COLUMN INDEX)
*                ALPHA STRINGS ( ROW INDEX )
*    END OF LOOPS
*
*
*. Input
      DIMENSION IOCOC(NOCTPA,NOCTPB),ISMOST(*)
      DIMENSION NSSOA(MAXSYM,NOCTPA),NSSOB(MAXSYM,NOCTPB)
      DIMENSION IBLTP(*)
*. output
      DIMENSION IOOS(NOCTPA,NOCTPB,MAXSYM)
      DIMENSION NOOS(NOCTPA,NOCTPB,MAXSYM)
*
      CALL ISETVC(IOOS,0,NOCTPA*NOCTPB*MAXSYM)
      CALL ISETVC(NOOS,0,NOCTPA*NOCTPB*MAXSYM)
C?    CALL ISETVC(ICBLTP,0,MAXSYM)
      NCOMB = 0
C?    WRITE(6,*) ' ZOOS: IDC  = ', IDC
      DO 100 IASYM = 1, MAXSYM
*
        IBSYM = ISMOST(IASYM)
        IF(IBSYM .EQ. 0 ) GOTO 100
*. Allowed combination symmetry block ?
        IF(IDC.NE.1.AND.IBLTP(IASYM).EQ.0) GOTO 100
*. Allowed occupation combinations
        DO  95 IAOCC = 1,NOCTPA
          IF(IBLTP(IASYM).EQ.1) THEN
            MXBOCC = NOCTPB
            IREST1 = 0
          ELSE
            MXBOCC = IAOCC
            IREST1 = 1
          END IF
          DO 90 IBOCC = 1, MXBOCC
*.Is this block allowed
            IF(IOCOC(IAOCC,IBOCC).EQ.1) THEN
              IOOS(IAOCC,IBOCC,IASYM) = NCOMB+1
              IF(IXPND.EQ.0 .AND. IREST1.EQ.1 .AND. IAOCC.EQ.IBOCC)THEN
                NCOMB = NCOMB
     &      +   (NSSOA(IASYM,IAOCC)+1)*NSSOB(IBSYM,IBOCC)/2
                NOOS(IAOCC,IBOCC,IASYM) =
     &          (NSSOA(IASYM,IAOCC)+1)*NSSOB(IBSYM,IBOCC)/2
              ELSE
                NCOMB = NCOMB
     &      +   NSSOA(IASYM,IAOCC)*NSSOB(IBSYM,IBOCC)
                NOOS(IAOCC,IBOCC,IASYM) =
     &          NSSOA(IASYM,IAOCC)*NSSOB(IBSYM,IBOCC)
              END IF
            END IF
C?      write(6,*) ' NOOS(IA,IB,ISM) ',NOOS(IAOCC,IBOCC,IASYM)
   90     CONTINUE
   95   CONTINUE
  100 CONTINUE
*
      NTEST = 0 
      IF ( NTEST .NE. 0 ) THEN
         WRITE(6,*) 
         WRITE(6,*) ' ==============='
         WRITE(6,*) ' ZOOS reporting '
         WRITE(6,*) ' ==============='
         WRITE(6,*) 
         WRITE(6,*) ' NSSOA, NSSOB ( input ) '
         CALL IWRTMA(NSSOA,MAXSYM,NOCTPA,MAXSYM,NOCTPA)
         CALL IWRTMA(NSSOB,MAXSYM,NOCTPB,MAXSYM,NOCTPB)
         WRITE(6,*) 
         WRITE(6,*) ' Number of combinations obtained ',NCOMB
         WRITE(6,*) ' Offsets for combination OOS blocks '
         DO 111 IASYM = 1,MAXSYM
           WRITE(6,'(A,I2)') '  Symmetry ',IASYM
           CALL IWRTMA(IOOS(1,1,IASYM),NOCTPA,NOCTPB,NOCTPA,NOCTPB)
  111    CONTINUE
         WRITE(6,*) ' Number of  combinations per OOS blocks '
         DO 112 IASYM = 1,MAXSYM
           WRITE(6,'(A,I2)') '  Symmetry ',IASYM
           CALL IWRTMA(NOOS(1,1,IASYM),NOCTPA,NOCTPB,NOCTPA,NOCTPB)
  112    CONTINUE
      END IF
*
      RETURN
      END
      SUBROUTINE ZSMCL(NSMST,NOCTP,NSSO,ISTSM,ISTCL)
*
* set symmetry and class arrays for strings
*
      INTEGER ISTSM(*),ISTCL(*),NSSO(NOCTP,NSMST)
*
      IOFF = 1
      DO 100 ISM = 1, NSMST
      DO 100 ICL = 1, NOCTP
        CALL  ISETVC(ISTSM(IOFF),ISM,NSSO(ICL,ISM))
        CALL  ISETVC(ISTCL(IOFF),ICL,NSSO(ICL,ISM))
        IOFF = IOFF + NSSO(ICL,ISM)
  100 CONTINUE
*
      RETURN
      END
      SUBROUTINE ZSTINF(IPRNT)
*
* Set up common block /STINF/ from information in /STINP/
*
*=========
* Input
*=========
* Information in /STINP/ and /ORBINP/
*
*======================
* Output ( in /STINF/ )
*======================
* ISTAC (MXPSTT,2): string type obtained by creating (ISTAC(ITYP,2))
*                    or annihilating (ISTAC(ITYP,1)) an electron
*                    from a string of type  ITYP . A zero indicates
*                    that this mapping is not included
*                    Only strings having the same ISTTP index are
*                    mapped
* NOCTYP(ITYP): Number of occupation classes for given type
*
*
* NSTFTP(ITYP): Number of strings of this type
*
* INUMAP(ITYP): Mapping of string type to next more general type
* INDMAP(ITYP): Mapping of string type to next more restricted type
*
*   / \           Zero order space                         |
*    |            Double excitations from reference space  |  Down
* Up |            single excitation from reference space   |
*    |            reference space                         \ /
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'strinp.inc'
*
      INCLUDE 'stinf.inc'
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
* ******************************************************************
* Mappings between strings with the same type ISTTP index , +/- 1 el
* ******************************************************************
      CALL ISETVC(ISTAC,0,2*MXPSTT)
C     DO 100 IAC = 1,2
C       IDELTA = (-1)**IAC
C       DO 90 ITYP = 1, NSTTYP
C         IISTTP = ISTTP(ITYP)
C          NEL = NELEC(ITYP) + IDELTA
C          JTYP = 0
C          DO 40 JJTYP = 1, NSTTYP
C            IF(NELEC(JJTYP).EQ.NEL.AND.ISTTP(JJTYP).EQ.IISTTP)
C    &       JTYP = JJTYP
C  40     CONTINUE
C         ISTAC(ITYP,IAC) = JTYP
C  90   CONTINUE
C 100 CONTINUE
      DO 90 ITYP = 1, NSTTYP-1
        IF(NELEC(ITYP+1).EQ.NELEC(ITYP)-1) THEN
          ISTAC(ITYP,1) = ITYP+1
          ISTAC(ITYP+1,2) = ITYP
        END IF
   90 CONTINUE
*
      IF(NTEST .NE. 0 ) THEN
        WRITE(6,*) ' Type - type mapping array ISTAC '
        WRITE(6,*) ' =============================== '
        CALL IWRTMA(ISTAC,NSTTYP,2,MXPSTT,2)
      END IF
* **************************************************
*. Number of occupation classes and strings per type
* **************************************************
      DO 200 ITYP = 1,NSTTYP
        NOCTYP(ITYP) = (MXRS1(ITYP)-MNRS1(ITYP)+1)
     &               * (MXRS3(ITYP)-MNRS3(ITYP)+1)
  200 CONTINUE
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Number of occupation classes per type '
        WRITE(6,*) ' ===================================== '
        CALL IWRTMA(NOCTYP,1,NSTTYP,1,NSTTYP)
      END IF
*
      DO 300 ITYP = 1, NSTTYP
        NSTFTP(ITYP) = NUMST3(NELEC(ITYP),NORB1,MNRS1(ITYP),MXRS1(ITYP),
     &                        NORB2,NORB3,MNRS3(ITYP),MXRS3(ITYP))
  300 CONTINUE
      IF(NTEST.NE.0) THEN
        WRITE(6,*) ' Number of strings per  type '
        WRITE(6,*) ' =========================== '
        CALL IWRTMA(NSTFTP,1,NSTTYP,1,NSTTYP)
      END IF
* *****************************************************************
*. Mappings between strings containing the same number of electrons
* *****************************************************************
      CALL ISETVC(INUMAP,0,MXPSTT)
      CALL ISETVC(INDMAP,0,MXPSTT)
*. Mapping to and from zero order space
*, Mapping from highest excitation in reference space to zero order space
*
*. Alpha strings
      NAEL = NELEC(IAZTP)
      IMXRFA = 0
      IF(IARTP(3,5).NE.0) THEN
        IMXRFA = IARTP(3,5)
      ELSE IF(IARTP(2,5).NE.0) THEN
        IMXRFA = IARTP(2,5)
      ELSE IF(IARTP(1,5).NE.0) THEN
        IMXRFA = IARTP(1,5)
      END IF
      IF(IMXRFA .NE. 0 ) THEN
        INUMAP(IMXRFA) = IAZTP
        IF(NAEL.GE.1) INUMAP(IMXRFA+1) = IAZTP+1
        IF(NAEL.GE.2) INUMAP(IMXRFA+2) = IAZTP+2
        INDMAP(IAZTP) =  IMXRFA   
        IF(NAEL.GE.1) INDMAP(IAZTP +1) = IMXRFA+1
        IF(NAEL.GE.2) INDMAP(IAZTP +2) = IMXRFA+2
      END IF
*. beta  strings
      NBEL = NELEC(IBZTP)
      IMXRFB = 0
      IF(IARTP(3,5).NE.0) THEN
        IMXRFB = IBRTP(3,5)
      ELSE IF(IARTP(2,5).NE.0) THEN
        IMXRFB = IBRTP(2,5)
      ELSE IF(IARTP(1,5).NE.0) THEN
        IMXRFB = IBRTP(1,5)
      END IF
      IF(IMXRFB.NE.0) THEN
        INUMAP(IMXRFB) = IBZTP
        IF(NBEL.GE.1) INUMAP(IMXRFB+1) = IBZTP+1
        IF(NBEL.GE.2) INUMAP(IMXRFB+2) = IBZTP+2
        INDMAP(IBZTP) =  IMXRFB   
        IF(NBEL.GE.1) INDMAP(IBZTP +1) = IMXRFB+1
        IF(NBEL.GE.2) INDMAP(IBZTP +2) = IMXRFB+2
      END IF
*
*. Up and down mappings of reference strings
*
      DO 450 IDEL = -4,2
      DO 430 IEX = 1,2
        IF(IARTP(IEX,IDEL+5).NE.0.AND.IARTP(IEX+1,IDEL+5).NE.0)THEN
          INUMAP(IARTP(IEX,IDEL+5)) = IARTP(IEX+1,IDEL+5)
          INDMAP(IARTP(IEX+1,IDEL+5)) = IARTP(IEX,IDEL+5)
        END IF
        IF(IBRTP(IEX,IDEL+5).NE.0.AND.IBRTP(IEX+1,IDEL+5).NE.0)THEN
          INUMAP(IBRTP(IEX,IDEL+5)) = IBRTP(IEX+1,IDEL+5)
          INDMAP(IBRTP(IEX+1,IDEL+5)) = IBRTP(IEX,IDEL+5)
        END IF
  430 CONTINUE
  450 CONTINUE
*
      IF(NTEST .NE. 0 ) THEN
        WRITE(6,*) ' Up mappings of string types '
        CALL IWRTMA(INUMAP,1,NSTTYP,1,NSTTYP)
        WRITE(6,*) ' Down mappings of string types '
        CALL IWRTMA(INDMAP,1,NSTTYP,1,NSTTYP)
      END IF
*
      RETURN
      END
      SUBROUTINE ZSYM1(NIRREP,IPRNT)
*
* Number of symmetries for d2h
* Symmetry connecting arrays
* ( trivial, written for compatibility with higher point groups)
*
      INTEGER SYMPRO(8,8)
      DATA  SYMPRO/1,2,3,4,5,6,7,8,
     &             2,1,4,3,6,5,8,7,
     &             3,4,1,2,7,8,5,6,
     &             4,3,2,1,8,7,6,5,
     &             5,6,7,8,1,2,3,4,
     &             6,5,8,7,2,1,4,3,
     &             7,8,5,6,3,4,1,2,
     &             8,7,6,5,4,3,2,1 /
C     COMMON/CSM/NSMSX,NSMDX,NSMST,NSMCI,ITSSX,ITSDX
      INCLUDE 'csm.inc'
*
C     PARAMETER ( MXPOBS = 20 )
      INCLUDE 'mxpdim.inc'
      INCLUDE 'csmprd.inc'
c      INTEGER ADASX,ASXAD,ADSXA,SXSXDX,SXDXSX
c      COMMON/CSMPRD/ADASX(MXPOBS,MXPOBS),ASXAD(MXPOBS,2*MXPOBS),
c     &              ADSXA(MXPOBS,2*MXPOBS),
c     &              SXSXDX(2*MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
 
      NSMSX = NIRREP
      NSMDX = NIRREP
      NSMST = NIRREP
      NSMCI = NIRREP
      NSMXT = NIRREP
      ITSSX = 1
      ITSDX = 1
      ITSXT = 1
 
*
C     COPMT2(AIN,AOUT,NINR,NINC,NOUTR,NOUTC,IZERO)
      CALL ICPMT2(SYMPRO,ADASX,8,8,MXPOBS,MXPOBS,1)
      CALL ICPMT2(SYMPRO,ADSXA,8,8,MXPOBS,2*MXPOBS,1)
      CALL ICPMT2(SYMPRO,ASXAD,8,8,MXPOBS,2*MXPOBS,1)
      CALL ICPMT2(SYMPRO,SXSXDX,8,8,2*MXPOBS,2*MXPOBS,1)
      CALL ICPMT2(SYMPRO,SXDXSX,8,8,2*MXPOBS,4*MXPOBS,1)
*
      RETURN
      END
      SUBROUTINE ZSYM2(IPRNT)
*
* Symmetry connecting arrays
*
* ======
*. Input
* ======
*
*./LUCINP/
C     INTEGER PNTGRP,CITYP,EXTSPC
C     PARAMETER(MXPIRR = 20)
C     PARAMETER(MXPOBS = 20 )
C     PARAMETE
C     COMMON/LUCINP/PNTGRP,NIRREP,NSMCMP,MAXML,MAXL,
C    &              INTSPC,EXTSPC,NRSSH(MXPIRR,3),
C    &              MNRS1R,MXRS1R,MNRS3R,MXRS3R,NACTEL,
C    &              NSMOB,NRS0SH(1,MXPIRR),NRS4SH(MXPR4T,MXPIRR),
C    &              MXR4TP, MXHR0,MXER4,
C    &              NINASH(MXPIRR),
C    &              INTXCI,NDELSH(MXPIRR),MNRS10,MXRS30
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
*./NONAB/
      LOGICAL INVCNT
      COMMON/NONAB/ INVCNT,NIG,NORASM(MXPOBS),
     &              MNMLOB,MXMLOB,NMLOB,
     &              MXMLST,MNMLST,NMLST,
     &              NMLSX ,MNMLSX,MXMLSX,
     &              MNMLCI,MXMLCI,NMLCI,
     &              MXMLDX,MNMLDX,NMLDX
*./CSM
C     COMMON/CSM/NSMSX,NSMDX,NSMST,NSMCI,ITSSX,ITSDX
      INCLUDE 'csm.inc'
*
* ======
*.Output
* ======
*
      INCLUDE 'csmprd.inc'
c*./CSMPRD/
c      INTEGER ADASX,ASXAD,ADSXA,SXSXDX,SXDXSX
c      COMMON/CSMPRD/ADASX(MXPOBS,MXPOBS),ASXAD(MXPOBS,2*MXPOBS),
c     &              ADSXA(MXPOBS,2*MXPOBS),
c     &              SXSXDX(2*MXPOBS,2*MXPOBS),SXDXSX(2*MXPOBS,4*MXPOBS)
*
**. ADASX,ASXAD,ADSXA
      CALL ISETVC(ADASX,0,MXPOBS**2)
      CALL ISETVC(ASXAD,0,2*MXPOBS**2)
      CALL ISETVC(ADSXA,0,2*MXPOBS**2)
*
      DO 100 ISM = 1, NSMOB
C       MLSM(IML,IPARI,ISM,TYPE,IWAY)
        CALL MLSM(IML,IPARI,ISM,'OB',2)
        DO 90 JSM = 1, NSMOB
          CALL MLSM(JML,JPARI,JSM,'OB',2)
*.a+ i a j symmetry
          IJML = IML - JML
          IF((IPARI.EQ.1.AND.JPARI.EQ.1).OR.
     &       (IPARI.EQ.2.AND.JPARI.EQ.2)) THEN
            IJPARI = 1
          ELSE
            IJPARI = 2
          END IF
          IJSM = (IJPARI-1)*NMLSX + IJML - MNMLSX + 1
          ADASX(ISM,JSM) = IJSM
          ASXAD(JSM,IJSM) = ISM
          ADSXA(ISM,IJSM) = JSM
   90   CONTINUE
  100 CONTINUE
*.SXSXDX,SXDXSX
      DO 200 ISX = 1, NSMSX
C       MLSM(IML,IPARI,ISM,TYPE,IWAY)
        CALL MLSM(IML,IPARI,ISX,'SX',2)
        DO 190 JSX = 1, NSMSX
          CALL MLSM(JML,JPARI,JSX,'SX',2)
          IF((IPARI.EQ.1.AND.JPARI.EQ.1).OR.
     &       (IPARI.EQ.2.AND.JPARI.EQ.2)) THEN
            IJPARI = 1
          ELSE
            IJPARI = 2
          END IF
          IJML = IML + JML
          IJSM = (IJPARI-1)*NMLDX+IJML - MNMLDX + 1
          SXSXDX(ISX,JSX) = IJSM
          SXDXSX(ISX,IJSM) = JSX
  190   CONTINUE
  200 CONTINUE
*
      NTEST = 0
      NTEST = MAX(NTEST,IPRNT)
      IF(NTEST.GE.10) THEN
         WRITE(6,*) ' ADASX '
         WRITE(6,*) ' ===== '
         CALL IWRTMA(ADASX,NSMOB,NSMOB,MXPOBS,MXPOBS)
         WRITE(6,*) ' ASXAD '
         WRITE(6,*) ' ===== '
         CALL IWRTMA(ASXAD,NSMOB,NSMSX,MXPOBS,2*MXPOBS)
         WRITE(6,*) ' ADSXA '
         WRITE(6,*) ' ===== '
         CALL IWRTMA(ADSXA,NSMOB,NSMSX,MXPOBS,2*MXPOBS)
         WRITE(6,*) ' SXSXDX'
         WRITE(6,*) ' ======'
         CALL IWRTMA(SXSXDX,NSMSX,NSMSX,2*MXPOBS,2*MXPOBS)
         WRITE(6,*) ' SXDXSX'
         WRITE(6,*) ' ======'
         CALL IWRTMA(SXDXSX,NSMSX,NSMDX,2*MXPOBS,4*MXPOBS)
      END IF
*
      RETURN
      END
      SUBROUTINE GETINCN(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,INTLST,IJKLOF,NSMOB,I2INDX,
     &                  ICOUL) 
*
* Obtain integrals 
*
*     ICOUL = 0: 
*                  XINT(IK,JL) = (IJ!KL)         for IXCHNG = 0
*                              = (IJ!KL)-(IL!KJ) for IXCHNG = 1
*
*     ICOUL = 1: 
*                  XINT(IJ,KL) = (IJ!KL)         for IXCHNG = 0
*                              = (IJ!KL)-(IL!KJ) for IXCHNG = 1
*
* Storing for ICOUL = 1 not working if IKSM or JLSM .ne. 0 
* 
*
* Version for integrals stored in INTLST
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*. Integral list
      Real * 8 Intlst(*)
      Dimension IJKLof(NsmOB,NsmOb,NsmOB)
*. Pair of orbital indeces ( symmetry ordered ) => address in symmetry packed 
*. matrix
      Dimension I2INDX(*)
*.Output
      DIMENSION XINT(*)
*. Local scratch      
      DIMENSION IJARR(MXPORB)
*. To get rid of annoying and incorrect compiler warnings
      IJRELKL = 0
      IBLOFF  = 0
      ILRELKJ = 0
      iOrb=NOBPTS(ITP,ISM)
      jOrb=NOBPTS(JTP,JSM)
      kOrb=NOBPTS(KTP,KSM)
      lOrb=NOBPTS(LTP,LSM)
*. Offsets relative to start of all orbitals, symmetry ordered 
      IOFF = IBSO(ISM)
      DO IITP = 1, ITP -1
        IOFF = IOFF + NOBPTS(IITP,ISM)
      END DO
*
      JOFF = IBSO(JSM)
      DO JJTP = 1, JTP -1
        JOFF = JOFF + NOBPTS(JJTP,JSM)
      END DO
*
      KOFF = IBSO(KSM)
      DO KKTP = 1, KTP -1
        KOFF = KOFF + NOBPTS(KKTP,KSM)
      END DO
*
      LOFF = IBSO(LSM)
      DO LLTP = 1, LTP -1
        LOFF = LOFF + NOBPTS(LLTP,LSM)
      END DO

*
*     Collect Coulomb terms
*
      ijblk = max(ism,jsm)*(max(ism,jsm)-1)/2 + min(ism,jsm)
      klblk = max(ksm,lsm)*(max(ksm,lsm)-1)/2 + min(ksm,lsm)
*
      IF(IJBLK.GT.KLBLK) THEN
       IJRELKL = 1
       IBLOFF=IJKLOF(MAX(ISM,JSM),MIN(ISM,JSM),MAX(KSM,LSM))
      ELSE IF (IJBLK.EQ.KLBLK) THEN
       IJRELKL = 0
       IBLOFF=IJKLOF(MAX(ISM,JSM),MIN(ISM,JSM),MAX(KSM,LSM))
      ELSE IF (IJBLK.LT.KLBLK) THEN
       IJRELKL = -1
       IBLOFF = IJKLOF(MAX(KSM,LSM),MIN(KSM,LSM),MAX(ISM,JSM))
      END IF
*
      itOrb=NTOOBS(iSm)
      jtOrb=NTOOBS(jSm)
      ktOrb=NTOOBS(kSm)
      ltOrb=NTOOBS(lSm)
*
      If(ISM.EQ.JSM) THEN
       IJPAIRS = ITORB*(ITORB+1)/2
      ELSE
       IJPAIRS = ITORB*JTORB
      END IF
*
      IF(KSM.EQ.LSM) THEN
        KLPAIRS = KTORB*(KTORB+1)/2
      ELSE
        KLPAIRS = KTORB*LTORB
      END IF
*
      iInt=0
      Do lJeppe=lOff,lOff+lOrb-1
        jMin=jOff
        If ( JLSM.ne.0 ) jMin=lJeppe
        Do jJeppe=jMin,jOff+jOrb-1
*
*
*. Set up array IJ*(IJ-1)/2 
          IF(IJRELKL.EQ.0) THEN 
            DO II = IOFF,IOFF+IORB-1
              IJ = I2INDX((JJEPPE-1)*NTOOB+II)
              IJARR(II) = IJ*(IJ-1)/2
            END DO
          END IF
*
          Do kJeppe=kOff,kOff+kOrb-1
            iMin = iOff
            kl = I2INDX(KJEPPE+(LJEPPE-1)*NTOOB)
            If(IKSM.ne.0) iMin = kJeppe
            IF(ICOUL.EQ.1)  THEN  
*. Address before integral (1,j!k,l)
                IINT = (LJEPPE-LOFF)*Jorb*Korb*Iorb
     &                + (KJEPPE-KOFF)*Jorb*Iorb
     &                + (JJEPPE-JOFF)*Iorb
            END IF
*
            IF(IJRELKL.EQ.1) THEN
*. Block (ISM JSM ! KSM LSM ) with (Ism,jsm) > (ksm,lsm)
              IJKL0 = IBLOFF-1+(kl-1)*ijPairs
              IJ0 = (JJEPPE-1)*NTOOB         
              Do iJeppe=iMin,iOff+iOrb-1
                  ijkl = ijkl0 + I2INDX(IJEPPE+IJ0)
                  iInt=iInt+1
                  Xint(iInt) = Intlst(ijkl)
              End Do
            END IF
*
*. block (ISM JSM !ISM JSM)
            IF(IJRELKL.EQ.0) THEN 
              IJ0 = (JJEPPE-1)*NTOOB         
              KLOFF = KL*(KL-1)/2
              IJKL0 = (KL-1)*IJPAIRS-KLOFF
              Do iJeppe=iMin,iOff+iOrb-1
                ij = I2INDX(IJEPPE+IJ0   )
                If ( ij.ge.kl ) Then
C                 ijkl=ij+(kl-1)*ijPairs-klOff
                  IJKL = IJKL0 + IJ
                Else
                  IJOFF = IJARR(IJEPPE)
                  ijkl=kl+(ij-1)*klPairs-ijOff
                End If
                iInt=iInt+1
                Xint(iInt) = Intlst(iblOff-1+ijkl)
              End Do
            END IF
*
*. Block (ISM JSM ! KSM LSM ) with (Ism,jsm) < (ksm,lsm)
            IF(IJRELKL.EQ.-1) THEN 
              ijkl0 = IBLOFF-1+KL - KLPAIRS
              IJ0 = (JJEPPE-1)*NTOOB         
              Do iJeppe=iMin,iOff+iOrb-1
                IJKL = IJKL0 + I2INDX(IJEPPE + IJ0)*KLPAIRS
                iInt=iInt+1
                Xint(iInt) = Intlst(ijkl)
              End Do
            END IF
*
          End Do
        End Do
      End Do
*
*     Collect Exchange terms
*
      If ( IXCHNG.ne.0 ) Then
*
      IF(ISM.EQ.LSM) THEN
       ILPAIRS = ITORB*(ITORB+1)/2
      ELSE
       ILPAIRS = ITORB*LTORB
      END IF
*
      IF(KSM.EQ.JSM) THEN
        KJPAIRS = KTORB*(KTORB+1)/2
      ELSE
        KJPAIRS = KTORB*JTORB
      END IF
*
        ilblk = max(ism,lsm)*(max(ism,lsm)-1)/2 + min(ism,lsm)
        kjblk = max(ksm,jsm)*(max(ksm,jsm)-1)/2 + min(ksm,jsm)
        IF(ILBLK.GT.KJBLK) THEN
          ILRELKJ = 1
          IBLOFF = IJKLOF(MAX(ISM,LSM),MIN(ISM,LSM),MAX(KSM,JSM))
        ELSE IF(ILBLK.EQ.KJBLK) THEN
          ILRELKJ = 0
          IBLOFF = IJKLOF(MAX(ISM,LSM),MIN(ISM,LSM),MAX(KSM,JSM))
        ELSE IF(ILBLK.LT.KJBLK) THEN
          ILRELKJ = -1
          IBLOFF = IJKLOF(MAX(KSM,JSM),MIN(KSM,JSM),MAX(ISM,LSM))
        END IF
*
        iInt=0
        Do lJeppe=lOff,lOff+lOrb-1
          jMin=jOff
          If ( JLSM.ne.0 ) jMin=lJeppe
*
          IF(ILRELKJ.EQ.0) THEN
           DO II = IOFF,IOFF+IORB-1
             IL = I2INDX(II+(LJEPPE-1)*NTOOB)
             IJARR(II) = IL*(IL-1)/2
           END DO
          END IF
*
          Do jJeppe=jMin,jOff+jOrb-1
            Do kJeppe=kOff,kOff+kOrb-1
              KJ = I2INDX(KJEPPE+(JJEPPE-1)*NTOOB)
              KJOFF = KJ*(KJ-1)/2
              iMin = iOff
*
              IF(ICOUL.EQ.1)  THEN
*. Address before integral (1,j!k,l)
                  IINT = (LJEPPE-LOFF)*Jorb*Korb*Iorb
     &                  + (KJEPPE-KOFF)*Jorb*Iorb
     &                  + (JJEPPE-JOFF)*Iorb
              END IF
*
              If(IKSM.ne.0) iMin = kJeppe
*
              IF(ILRELKJ.EQ.1) THEN
                ILKJ0 = IBLOFF-1+( kj-1)*ilpairs
                IL0 = (LJEPPE-1)*NTOOB 
                Do iJeppe=iMin,iOff+iOrb-1
                  ILKJ = ILKJ0 + I2INDX(IJEPPE + IL0)
                  iInt=iInt+1
                  XInt(iInt)=XInt(iInt)-Intlst(ilkj)
                End Do
              END IF
*
              IF(ILRELKJ.EQ.0) THEN
                IL0 = (LJEPPE-1)*NTOOB 
                ILKJ0 = (kj-1)*ilPairs-kjOff
                Do iJeppe=iMin,iOff+iOrb-1
                  IL = I2INDX(IJEPPE + IL0 )
                  If ( il.ge.kj ) Then
C                     ilkj=il+(kj-1)*ilPairs-kjOff
                      ILKJ = IL + ILKJ0
                    Else
                      ILOFF = IJARR(IJEPPE)
                      ilkj=kj+(il-1)*kjPairs-ilOff
                    End If
                  iInt=iInt+1
                  XInt(iInt)=XInt(iInt)-Intlst(iBLoff-1+ilkj)
                End Do
              END IF
*
              IF(ILRELKJ.EQ.-1) THEN
                ILKJ0 = IBLOFF-1+KJ-KJPAIRS
                IL0 = (LJEPPE-1)*NTOOB
                Do iJeppe=iMin,iOff+iOrb-1
                  ILKJ = ILKJ0 + I2INDX(IJEPPE+ IL0)*KJPAIRS
                  iInt=iInt+1
                  XInt(iInt)=XInt(iInt)-Intlst(ilkj)
                End Do
              END IF
*
            End Do
          End Do
        End Do
      End If
*
      Return
      End
      SUBROUTINE RDVEC(LU,NSYM,NBAS,NORB,CMO,OCC,LOCC,TITLE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION NBAS(NSYM),NORB(NSYM),CMO(*),OCC(*)
      CHARACTER*(*) TITLE
      CHARACTER LINE*80,FMT*40
      LOGICAL SET
      FMT='(4E18.12)'
      REWIND (LU)
      KCMO  = 0
      NDIV  = 4
      TITLE = ' '
      SET   = .FALSE.
      DO 100 ISYM=1,NSYM
         DO 110 IORB=1,NORB(ISYM)
            DO 111 IBAS=1,NBAS(ISYM),NDIV
112            READ(LU,'(A80)',END=999,ERR=999) LINE
               IF(LINE(1:1).EQ.'*') THEN
                  IF(.NOT. SET) THEN
                     TITLE=LINE
                     SET=.TRUE.
                  END IF
                  GOTO 112
               END IF
               READ(LINE,FMT)
     &             (CMO(I+KCMO),I=IBAS,MIN(IBAS+3,NBAS(ISYM)))
111         CONTINUE
            KCMO=KCMO+NBAS(ISYM)
110      CONTINUE
100   CONTINUE
      IF(LOCC.EQ.0) RETURN
      KOCC=0
      DO 200 ISYM=1,NSYM
         DO 210 IORB=1,NORB(ISYM),NDIV
212         READ(LU,'(A80)',END=999,ERR=999) LINE
            IF(LINE(1:1).EQ.'*') THEN
               IF(.NOT. SET) THEN
                  TITLE=LINE
                  SET=.TRUE.
               END IF
               GOTO 212
            END IF
            READ(LINE,FMT) (OCC(I+KOCC),I=IORB,MIN(IORB+3,NORB(ISYM)))
210      CONTINUE
         KOCC=KOCC+NORB(ISYM)
200   CONTINUE
      RETURN
999   CONTINUE
      WRITE(*,*) '* ERROR IN RDVEC WHILE READING VECTOR SOURCE FILE *'
      STOP 20
      END
      SUBROUTINE WRVEC(LU,NSYM,NBAS,NORB,CMO,OCC,LOCC,TITLE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION NBAS(NSYM),NORB(NSYM),CMO(*),OCC(*)
      CHARACTER*(*) TITLE
C     CHARACTER LINE*80,FMT*40
      CHARACTER FMT*40
      FMT='(4E18.12)'
      REWIND (LU)
      KCMO  = 0
      NDIV  = 4
      IF(TITLE(1:1).NE.'*') TITLE='*'//TITLE
      WRITE(LU,'(A)') TITLE
      DO 100 ISYM=1,NSYM
         DO 110 IORB=1,NORB(ISYM)
            WRITE(LU,'(A,2I5)') '* ORBITAL',ISYM,IORB
            DO 111 IBAS=1,NBAS(ISYM),NDIV
               WRITE(LU,FMT) (CMO(I+KCMO),I=IBAS,MIN(IBAS+3,NBAS(ISYM)))
111         CONTINUE
            KCMO=KCMO+NBAS(ISYM)
110      CONTINUE
100   CONTINUE
      IF(LOCC.EQ.0) RETURN
      WRITE(LU,'(A,2I5,A)') '* OCCUPATION NUMBERS'
      KOCC=0
      DO 200 ISYM=1,NSYM
         DO 210 IORB=1,NORB(ISYM),NDIV
            WRITE(LU,FMT) (OCC(I+KOCC),I=IORB,MIN(IORB+3,NORB(ISYM)))
210      CONTINUE
         KOCC=KOCC+NORB(ISYM)
200   CONTINUE
      RETURN
      END
      SUBROUTINE SETINT_FUSK(XINT1,XINT2,NINT1,NINT2)
*
* Fusk definition of one- and two- electron integrals
*
* Jeppe Olsen, May 2003
*
*. Output
      DIMENSION XINT1(NINT1),XINT2(NINT2)
*
      DO INT = 1, NINT1
        XINT1(INT) = FLOAT(INT)
      END DO
      DO INT = 1, NINT2
        XINT2(INT) = FLOAT(INT)
      END DO
*
      RETURN
      END
      SUBROUTINE COP_CHARVEC(CHAR_IN,CHAR_OUT,NCHAR)
*
* Copy between two char vectors
*
      CHARACTER*72 CHAR_IN, CHAR_OUT
*
      DO JCHAR = 1, NCHAR
        CHAR_OUT(JCHAR:JCHAR) = CHAR_IN(JCHAR:JCHAR)
      END DO
*
      RETURN
      END 
      SUBROUTINE REO_INT
*
* Master routine for reordering orbitals within given symmetry
*
* Jeppe Olsen, June 2002
*
*
c      INCLUDE 'implicit.inc'
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'glbbas.inc'
*
*. Obtain reorder array: IREO(N) is orig number of new orbital N
*
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'REO_IN')
*
      CALL MEMMAN(KLIREO_ORB,NTOOB,'ADDL  ',1,'REO_OR')
      CALL SET_REO_ORB_ARRAY(WORK(KLIREO_ORB))
*
* Reorder two-electron integral matrix
*
      CALL MEMMAN(KLINT2,NINT2,'ADDL  ',2,'LINT2 ')
C     REO_2INT(XIN,XOUT,IREO_ORB,IJKLOF)
      CALL REO_2INT(WORK(KINT2),WORK(KLINT2),WORK(KLIREO_ORB),
     &              WORK(KPINT2))
      CALL COPVEC(WORK(KLINT2),WORK(KINT2),NINT2)
*
* Reorder one-electron integrals 
*
      CALL MEMMAN(KLINT1,NINT1,'ADDL  ',2,'LINT1 ')
C     REO_1EMAT(HIN,HOUT,IREO_ORB,IHSM)
      CALL REO_1EMAT(WORK(KINT1),WORK(KLINT1),WORK(KLIREO_ORB),
     &               1)
      CALL COPVEC(WORK(KLINT1),WORK(KINT1),NINT1)
*
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'REO_IN')
      RETURN
      END
      SUBROUTINE SET_REO_ORB_ARRAY(IREO_ORB_ARRAY)
*
* Set up array giving reordering of orbitals
*.IREO_ORB_ARRAY(N) is orig number of new orbital N
*
* Jeppe Olsen, June 2002
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'crun.inc'
*. Output
      INTEGER IREO_ORB_ARRAY(*)
*
C          ISTVC2(IVEC,IBASE,IFACT,NDIM)
      CALL ISTVC2(IREO_ORB_ARRAY,0,1,NTOOB)

      DO ISWITCH = 1, NSWITCH
        ISM = IREO_ORB(1,ISWITCH)
        IOLD = IREO_ORB(2,ISWITCH)
        INEW = IREO_ORB(3,ISWITCH)
        IOFF = 1
        DO JSM = 1, ISM-1
          IOFF = IOFF + NTOOBS(JSM)
        END DO
        IREO_ORB_ARRAY(IOFF-1+IOLD) = IOFF-1+INEW
        IREO_ORB_ARRAY(IOFF-1+INEW) = IOFF-1+IOLD
      END DO
*
      NTEST = 000
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Array for reordering orbitals , new => orig '
        CALL IWRTMA(IREO_ORB_ARRAY,1,NTOOB,1,NTOOB)
      END IF
*
      RETURN
      END
      SUBROUTINE REO_2INT(XIN,XOUT,IREO_ORB,IJKLOF)
*
* Reorder two-electron integrals according to reorder array IREO_ORB
* That allows reordering of orbitals of the same symmetry
*
* Jeppe Olsen, June 2002, In sunny Helsingfors
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'multd2h.inc'
*. Start of given symmetry block
      INTEGER IJKLOF(NSMOB,NSMOB,NSMOB)
*. Input integrals
      DIMENSION XIN(*)
*. And output integrals
      DIMENSION XOUT(*)
*
      NTEST = 0
      DO ISM = 1, NSMOB
        DO JSM = 1, ISM 
          IJSM = MULTD2H(ISM,JSM)
          DO KSM = 1, ISM
            IJKSM = MULTD2H(IJSM,KSM)
            IH2SM = 1
            LSM = MULTD2H(IJKSM,IH2SM)
            IF(KSM.EQ.ISM) THEN
              LSM_MAX = JSM
            ELSE
              LSM_MAX = KSM
            END IF
            IF(LSM.LE.LSM_MAX) THEN
*. Offset to integral block
              IBLK_OFF = IJKLOF(ISM,JSM,KSM)
              IF(NTEST.GE.10) THEN
                WRITE(6,*) ' IBLK_OFF = ', IBLK_OFF
                WRITE(6,*) ' ISM, JSM, KSM, LSM = ',
     &                       ISM, JSM, KSM, LSM
              END IF
              CALL REO_2INT_BLK(XIN(IBLK_OFF),ISM,JSM,KSM,LSM,
     &             IREO_ORB,IBLK_OFF,XOUT(IBLK_OFF))
            END IF  
          END DO
        END DO
      END DO
*
      RETURN
      END 
      SUBROUTINE REO_2INT_BLK(XIN,ISM,JSM,KSM,LSM,IREO_ORB,IJKLOFF,
     &                        XOUT)
*
* Reorder a complete symmetry block  of integrals, 
* according to orbital reorder array * IREO_ORB
*
* On input ISM gt JSM, KSM ge LSM, IJSM ge KLSM
*
* XIN and XOUT are symmetryblocks, and IJKLOFF is start of this 
* symmetry block in complete list
*
* Jeppe Olsen, June 2002
*
      IMPLICIT REAL*8(A-H,O-Z)
*
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
*. Specific input
      DIMENSION XIN(*)
      INTEGER IREO_ORB(*)
*. Output
      DIMENSION XOUT(*)
*
      NTEST = 000
*
*. Offset and number of integrals
*
      NI = NTOOBS(ISM)
      NJ = NTOOBS(JSM)
      NK = NTOOBS(KSM)
      NL = NTOOBS(LSM)
*. Offset to start of orbitalsof given symmetry 
      IOFF = 1 + IELSUM(NTOOBS,ISM-1) 
      JOFF = 1 + IELSUM(NTOOBS,JSM-1) 
      KOFF = 1 + IELSUM(NTOOBS,KSM-1) 
      LOFF = 1 + IELSUM(NTOOBS,LSM-1) 
*
      IJKL = 0
      DO K = 1, NK
        KOLD = IREO_ORB(KOFF-1+K) 
        IF(KSM.EQ.LSM) THEN
         LMAX = K
        ELSE
         LMAX = NL
        END IF
        DO L = 1, LMAX
         LOLD = IREO_ORB(LOFF-1+L) 
*. We now have K and L, the loops over ij depends upon 
*. whether ISM = JSM, as this condition implies that inner loop is 
*. j in accordiance with the standard def of IJ
         IF(ISM.EQ.JSM) THEN
*
           IF(ISM.EQ.KSM.AND.JSM.EQ.LSM) THEN
             IMIN = K
           ELSE 
             IMIN = 1
           END IF
           DO I = IMIN,NI
             IOLD = IREO_ORB(IOFF-1+I) 
             IF(ISM.EQ.JSM) THEN
               JMAX = I
             ELSE 
               JMAX = NJ
             END IF
             IF(ISM.EQ.KSM.AND.JSM.EQ.LSM.AND.I.EQ.K) THEN
               JMIN = L
             ELSE
               JMIN = 1
             END IF
             DO J = JMIN, JMAX
*. Address of reordered integral w.r.t start of symmetryblock
              JOLD = IREO_ORB(J+JOFF-1)
              IJKL_OLD = I2EAD(IOLD,JOLD,KOLD,LOLD)-IJKLOFF+1
              IJKL = IJKL + 1
              IF(NTEST.GE.1000) THEN
                WRITE(6,*) ' I, J, K, L, IJKL, IJKL_OLD = ',
     &                       I, J, K, L, IJKL, IJKL_OLD 
              END IF
              XOUT(IJKL) = XIN(IJKL_OLD)
             END DO
           END DO
         ELSE 
*. ISM is .gt. JSM, so normal ordering of IJ
          IF(ISM.EQ.KSM.AND.JSM.EQ.LSM) THEN
            IMIN = K
          ELSE 
            IMIN = 1
          END IF
          DO I = IMIN, NI
            IOLD = IREO_ORB(IOFF-1+I) 
            IF(ISM.EQ.KSM.AND.JSM.EQ.LSM.AND.I.EQ.K) THEN
             JMIN = L
            ELSE
             JMIN = 1
            END IF
            DO J = JMIN, NJ
              JOLD = IREO_ORB(J+JOFF-1)
              IJKL_OLD = I2EAD(IOLD,JOLD,KOLD,LOLD)-IJKLOFF+1
              IJKL = IJKL + 1
              IF(NTEST.GE.1000) THEN
                WRITE(6,*) ' I, J, K, L, IJKL, IJKL_OLD = ',
     &                       I, J, K, L, IJKL, IJKL_OLD 
              END IF
              XOUT(IJKL) = XIN(IJKL_OLD)
            END DO
          END DO
        END IF
*       ^ End of Switching around between ISM = JSM and ISM > JSM
      END DO
      END DO
*    ^ End of loop over K and L
*
      RETURN
      END
      SUBROUTINE REO_1EMAT(HIN,HOUT,IREO_ORB,IHSM)
*
* Reorder One-electron integral matrix according with 
*.reorder matrix IREO_ORB. Matrix is assumed to be packed
*
* Jeppe Olsen, June 2002, Sunny Helsinki
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'multd2h.inc'
      INCLUDE 'orbinp.inc'
*. Input
      DIMENSION HIN(*)
      INTEGER IREO_ORB(*)
*. Output
      DIMENSION HOUT(*)
*
      IJ = 0
      DO IRSM = 1, NSMOB
        ICSM = MULTD2H(IRSM,IHSM)
*
        IROFF = IBSO(IRSM)
        ICOFF = IBSO(ICSM)
*
        NR = NTOOBS(IRSM)
        NC = NTOOBS(ICSM)
*. Offset to this symmetryblock
        IBLOFF = IJ + 1
        IF(IRSM.EQ.ICSM) THEN
*. Symmetric block, packed as lower triangular
*. Loop over indeces of new matrix block
*. Offset to this symmetry block
          DO I = 1, NR
          DO J = 1, I
            IJ = IJ + 1
            I_OLD = IREO_ORB(IROFF-1+I)-IROFF+1
            J_OLD = IREO_ORB(ICOFF-1+J)-ICOFF+1
            IJ = IBLOFF-1+I*(I-1)/2+J
            IJ_OLD = IBLOFF - 1  
     &             + MAX(I_OLD,J_OLD)*(MAX(I_OLD,J_OLD)-1)/2
     &             + MIN(I_OLD,J_OLD)
            HOUT(IJ) = HIN(IJ_OLD)
          END DO
          END DO
        ELSE IF (IRSM.GT.ICSM) THEN
*. Complete symmetryblock is present
          DO I = 1, NR
          DO J = 1, NC
            IJ = IJ + 1
            I_OLD = IREO_ORB(IROFF-1+I)-IROFF+1
            J_OLD = IREO_ORB(ICOFF-1+J)-ICOFF+1
            IJ = IBLOFF-1+(J-1)*NR + I
            IJ_OLD = IBLOFF-1+(J_OLD-1)*NR + I_OLD
            HOUT(IJ) = HIN(IJ_OLD)
          END DO
          END DO
        END IF
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN  
        WRITE(6,*) ' REO_1EMAT: Input one-electron matrix '
C            PRHONE(H,NFUNC,IHSM,NSM,IPACK)
        CALL PRHONE(HIN,NTOOBS,IHSM,NSMOB,1)
        WRITE(6,*) ' REO_1EMAT: Output one-electron matrix '
        CALL PRHONE(HOUT,NTOOBS,IHSM,NSMOB,1)
      END IF
*
      RETURN 
      END
      SUBROUTINE GET_NB_FOR_PH(IHP,ISMOB,IB,NUM)
*
* Obtain start and number of orbitals for orbitals 
* of given sym and H/P
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cgas.inc'
*
      IF(IHP.EQ.1) THEN
*. Hole orbitals
        IB = IBSO(ISMOB)
        NUM = 0
        DO ITP = 1, N_HOLE_ORBSPACE 
         NUM = NUM + NOBPTS(ITP,ISMOB)
        END DO
      ELSE
*. Particle orbitals
        IB = IBSO(ISMOB)
        DO ITP = 1, N_HOLE_ORBSPACE
         IB = IB +  NOBPTS(ITP,ISMOB)
        END DO
        NUM = 0
        DO ITP =  N_HOLE_ORBSPACE+1, NGAS
         NUM = NUM + NOBPTS(ITP,ISMOB)
        END DO
      END IF
*
      NTEST = 000
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Input IHP, ISMOB = ', IHP,ISMOB
        WRITE(6,*) ' Output: IB, NUM ', IB,NUM
      END IF
*
      RETURN
      END
      SUBROUTINE GET_QD_INTS(H1,H2,INTP1)
*
*  Read in one- and two- body integrals defining qdot problem
*
* Jeppe Olsen, May 1999 
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'multd2h.inc'
*. Pointer to symmetric packed symmetric one-electronintegrals
      DIMENSION INTP1(*)
*. Output
      DIMENSION H1(*), H2(*)
*
* One-electron integrals: First H(hh), then H(pp)
*
      LUH1 = 90
      REWIND LUH1
*. Loop over particle and hole spaces      
      DO IHP = 1, 2
        DO ISMOB = 1, NSMOB
          CALL GET_NB_FOR_PH(IHP,ISMOB,IB,NI)
          WRITE(6,*) ' ISMOB, NI = ', ISMOB,NI 
          ISOFF = INTP1(ISMOB)
          DO I = 1, NI
          DO J = 1, I
            II = I + IB - IBSO(ISMOB)
            JJ = J + IB - IBSO(ISMOB)
            IJ = ISOFF - 1 + II*(II-1)/2+JJ 
            WRITE(6,*) ' IHP, ISMOB, I, J = ', IHP, ISMOB, I,J 
            READ(LUH1,*) H1(IJ)
            WRITE(6,*) ' IJ and H1(IJ) ', IJ, H1(IJ)
          END DO
          END DO
        END DO   
      END DO
*
* Two body integrals
*
*
      LUHHHH = 81
      LUHHPP = 82
      LUPPPP = 83
*
      DO ICASE = 1, 3
* ICASE = 1 => hhhh
*       = 2 => hhpp
*       = 3 => pppp
*
        WRITE(6,*) ' ICASE = ', ICASE
        IF(ICASE.EQ.1) THEN
          LUH = LUHHHH
        ELSE IF(ICASE.EQ.2) THEN
          LUH = LUHHPP
        ELSE 
          LUH = LUPPPP
        END IF
        REWIND LUH
*. Loop structure over symmetry blocks from TRA2 
        DO ISYM = 1, NSMOB
         DO JSYM = 1, NSMOB
          DO KSYM = 1, NSMOB
            IJSYM = MULTD2H(ISYM,JSYM)
            IJKSYM = MULTD2H(IJSYM,KSYM)
            INTSYM = 1
            LSYM = MULTD2H(IJKSYM,INTSYM)
*
C?            WRITE(6,'(A,4I4)') 
C?   &        ' ISYM, JSYM, KSYM, LSYM',ISYM,JSYM,KSYM,LSYM
*. Read in integrals as  (I,J,K,L)
              IF(ICASE.EQ.1.OR.ICASE.EQ.2) THEN
*. IJ corresponds to hole spaces
                IHP = 1
                JHP = 1
              ELSE
*. IJ corresponds to particle spaces
                IHP = 2
                JHP = 2
              END IF
*
              IF(ICASE.EQ.1) THEN
*. Kl corresponds to hole spaces
                KHP = 1
                LHP = 1
              ELSE
*. kl corresponds to particle spaces
                KHP = 2
                LHP = 2
              END IF
              CALL GET_NB_FOR_PH(IHP,ISYM,IB,NI)
              CALL GET_NB_FOR_PH(JHP,JSYM,JB,NJ)
              CALL GET_NB_FOR_PH(KHP,KSYM,KB,NK)
              CALL GET_NB_FOR_PH(LHP,LSYM,LB,NL)
C             GET_NB_FOR_PH(IHP,ISMOB,IBASE,NUM)
              DO L = 1, NL
              DO K = 1, NK
              DO J = 1, NJ
              DO I = 1, NI
C               WRITE(6,*) ' I,J,K,L', I,J,K,L
                READ(LUH,*) XINT
C               WRITE(6,*)  'XINT = ', XINT
                II = IB-1+I
                JJ = JB-1+J
                KK = KB-1+K
                LL = LB-1+L
                IJKL= I2EAD(II,JJ,KK,LL)
C?              WRITE(6,'(A,4I3,I9,E15.8)') 
C?   &          ' I,J,K,L,IJKL,XINT', II,JJ,KK,LL,IJKL,XINT
                H2(IJKL) = XINT
              END DO
              END DO
              END DO
              END DO
*.            ^ End of loop over orbitals belonging to given sym
          END DO
        END DO
      END DO
*.    ^ End of loop over symmetries
      END DO
*.    ^ End of loop over Cases(integral types )
*
      RETURN
      END
      LOGICAL FUNCTION FNDLAB(A,LU)
C
C 26-May-1985 hjaaj -- logical function version of SEARCH
C 16-Jun-1986 hjaaj -- changed to CHARACTER*8 from REAL*8
C
      CHARACTER*8 A, B(4), C
      DATA C/'********'/
    1 READ(LU,END=3,ERR=6)B
C?    WRITE(6,'(4A8)') ' Label from FNDLAB:',B(1),B(2),B(3),B(4)
      IF (B(1).NE.C) GO TO 1
      IF (B(4).NE.A) GO TO 1
      FNDLAB = .TRUE.
      GO TO 10
C
    6 CONTINUE
      GO TO 8
    3 CONTINUE
C
    8 FNDLAB = .FALSE.
C
   10 RETURN
      END
      SUBROUTINE INFSIR(IWRK,WRK,LWRK)
*
*. Required space in WRK : LWRK = NBAS + 7*NBAS**2
*. Required space in IWRK: NBAS
C
C     Written by Henrik Koch 06-09-89
*     Last modification: Jeppe Olsen, 2 sept 98.
*                         Jeppe Olsen, June 10: read of labels added
*                                      
C
C     Purpose: Reads in information to interface to R. Harrisons
C               Full CI program and to the subsequent polarization
C               propagator calculation.
C     Argument list:
C
      INCLUDE 'wrkspc.inc'
      INCLUDE 'irat.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      PARAMETER (LUINP = 17)
      PARAMETER ( ITAP30 = 16, LUPRI = 6, LUCMDS=17)
      PARAMETER ( LURSP   = 18 , LUONE = 19, LUW6 = 6)
      PARAMETER ( LBUF = 600 )
      DIMENSION BUF(LBUF), IBUF(LBUF)
C     DIMENSION TITLE(24),NDEG(8),NBAS(8),NOCC(8),NEJBAS(8)
      DIMENSION TITLE(24),NDEG(8),        NOCC(8),NEJBAS(8)
      DIMENSION IWRK(*),WRK(LWRK)
      CHARACTER*72 TITMOL(2)
*
      LOGICAL LPPOP
C     LOGICAL FNDLAB
      COMMON /CIPOL / NBAST, NNBAST, MORB(8), NMORBT, NORB(8),NNORB(8),
     *                NSYMHF, LBINTM, LPPOP(3,3), LSYMOP(3,3),
     *                NCMOT, NNORBX, NLAMDA(8), LUEGVC,NORBT,ISAT(128)
*
      CHARACTER*4 AO_CENT, AO_TYPE
      COMMON/AOLABELS/AO_CENT(MXPORB),AO_TYPE(MXPORB)
*. Local scratch
      DIMENSION ITMP(2*MXPORB)
     
      common /nuclear/ enenuc
*
      NTEST = 000
C
C
C     Read information on file AONEINT from HERMIT.
C
      ITAP34 = 66
      OPEN (ITAP34,STATUS='OLD',FORM='UNFORMATTED',FILE='AOONEINT')
      REWIND ITAP34
*. It seems like AOONEINT has changed so(Jan 11)
      INEW_OR_OLD = 1
      IF(INEW_OR_OLD.EQ.2) THEN
        READ (ITAP34) TITLE,NST,(NDEG(I),I=1,NST),ENUC
      ELSE
        READ(ITAP34) TITMOL
        READ (ITAP34) NST,(NDEG(I),I=1,NST),ENUC
      END IF
*. Labels of Atomic orbitals
      CALL MOLLAB('SYMINPUT',ITAP34,6)
      READ (ITAP34) NBT,(ITMP(I),I=1,2*NBT)
*. Transfer labels to character form
      DO I = 1, NBT
        WRITE(AO_CENT(I),'(A4)') ITMP(I)
        WRITE(AO_TYPE(I),'(A4)') ITMP(I+NBT)
        IF(NTEST.GE.100) WRITE(6,'(A, I4,2X, 2A4)') 
     &  ' I, AO_CENT(I),  AO_TYPE(I) = ', I, AO_CENT(I),AO_TYPE(I)
      END DO
*
      CLOSE(ITAP34,STATUS='KEEP')
      CORE = ENUC
      enenuc = enuc
      IPRINT = 0
C
      NSYMHF = 0
      MXCOEF = 0
      NBFAO  = 0
      DO 100 I = 1,NST
         IF (NDEG(I) .GT. 0) THEN
            NBFAO  = NBFAO  + NDEG(I)
            NSYMHF = NSYMHF + 1
            MXCOEF = MXCOEF + NDEG(I)*NDEG(I)
         END IF
  100 CONTINUE
      IF(INEW_OR_OLD.EQ.2) THEN
      WRITE(6,'(//A,2(/12A6)/)')
     *   ' Molecule title from basis set input:',(TITLE(I),I=1,24)
      ELSE
      WRITE(6,'(//A,2(/A72)/)')
     *   ' Molecule title from basis set input:',TITMOL(1),TITMOL(2)
      END IF
     
C?    WRITE(6,*) 'NBFAO : ',NBFAO
C?    WRITE(6,*) 'NSYMHF: ',NSYMHF
C
C     Read information on file SIRIFC written from SIRIUS.
C
      OPEN(ITAP30,STATUS='OLD',FORM='UNFORMATTED',FILE='SIRIFC')
      REWIND ITAP30
      CALL MOLLAB('TRCCINT ',ITAP30,6)
      READ (ITAP30) NSYMHF,NORBT,NBAST,NCMOT,(NOCC(I),I=1,NSYMHF),
     *              (NLAMDA(I),I=1,NSYMHF),(NORB(I),I=1,NSYMHF),
     *              POTNUC,EMCSCF
      CALL ICOPVE(NLAMDA,NEJBAS,8)
C. Memory in IWRK
      KEIGSY = 1
*. Memory in WORK
      KEIGVL = 1
      KEIGVC = KEIGVL + NORBT 
C
      READ (ITAP30) (WRK(KEIGVL+I-1),I=1,NORBT),
     *              (IWRK(KEIGSY+I-1),I=1,NORBT)
      READ (ITAP30) (WRK(KEIGVC+I-1),I=1,NCMOT)
      IF(NTEST.GE.10) WRITE(6,*) ' NOMOFL = ', NOMOFL
COLD  IF(NOMOFL.EQ.0) THEN
COLD    CALL COPVEC(WRK(KEIGVC),WORK(KMOAOIN),NCMOT)
        IF(NTEST.GE.10) THEN
         WRITE(6,*) ' MO-AO transformation matrix in INFSIR '
         CALL PRINT_CMOAO(WRK(KEIGVC))
        END IF
COLD  END IF
    
      CLOSE(ITAP30,STATUS='KEEP')
C
      LUEGVC = 60
      OPEN (LUEGVC,STATUS='UNKNOWN',FORM='UNFORMATTED',
     *      FILE='MOEIGVC')
      WRITE(LUEGVC) (WRK(KEIGVC+I-1),I=1,NCMOT)
      CLOSE (LUEGVC,STATUS='KEEP')
C
C     Check information from AONEINT and SIRGEOM.
C
      I_DO_CHECK = 0
      IF(I_DO_CHECK.EQ.0) THEN
        WRITE(6,*) ' Warning: No checks of consistency '
        WRITE(6,*) ' Between AOONEINT and SIRIFC '
      ELSE
      IF ((MXCOEF .NE. NCMOT) .OR. (NBFAO  .NE. NORBT) .OR.
     *    (NBAST  .NE. NBFAO) .OR. (ENUC   .NE. POTNUC)) THEN
         WRITE(LUPRI,*) 'Inconsistency error between AONEINT and'
         WRITE(LUPRI,*) 'SIRGEOM'
         WRITE(LUPRI,*) 'MXCOEF AND NCMOT',MXCOEF,NCMOT
         WRITE(LUPRI,*) 'NBFAO  AND NORBT',NBFAO,NORBT
         WRITE(LUPRI,*) 'NSYMHF AND NSYM',NSYMHF,NSYM
         WRITE(LUPRI,*) 'ENUC   AND POTNUC',ENUC,POTNUC
         STOP 'INCONSISTENCY ERROR IN LOAD'
      ELSE
         WRITE(LUPRI,*) 'Input from AONEINT and SIRGEOM was found'
         WRITE(LUPRI,*) 'to be ok, and we thus proceed.'
      ENDIF
      END IF
C
      IF (IPRINT .GT. 10) THEN
         DO 120 I = 1,NORBT
            WRITE(LUPRI,'(/A)') 'Orbital number, symmetry and energy'
            WRITE(LUPRI,'(A/)') '-----------------------------------'
            WRITE(LUPRI,'(I3,5X,I1,5X,F16.6)')
     *           I,IWRK(KEIGSY+I-1),WRK(KEIGVL+I-1)
  120    CONTINUE
      ENDIF
C
      IF (IPRINT .GT. 2) THEN
         WRITE(LUPRI,*) 'Nuclear repulsion energy: ',POTNUC
         WRITE(LUPRI,*) 'Total SCF energy        : ',EMCSCF
      ENDIF
C
      KONEEL = KEIGVC + NBAST*NBAST
      KMOONE = KONEEL + NBAST*NBAST
      KSCR1  = KMOONE + NBAST*NBAST
      KEND   = KSCR1  + 3*NBAST*NBAST
      IF ( KEND-1.GT. LWRK ) THEN
         WRITE(6,*) ' Insufficient space in INFSIR '
         WRITE(6,*) ' Required and allocated ', KEND-1,LWRK
         WRITE(6,*) ' NORBT, NBAST = ', NORBT, NBAST
         STOP 'Insufficient space in INFSIR'
      ENDIF
C
C     ********************************************************
C     * Read one-electron integrals and transform to MO-basis*
C     ********************************************************
C
      NCOEF_MO_MO = 0
      NCOEF_1EL = 0
      DO ISYM = 1, NSYMHF
        NCOEF_MO_MO = NCOEF_MO_MO + NORB(ISYM)**2
        NCOEF_1EL = NCOEF_1EL + NORB(ISYM)*(NORB(ISYM)+1)/2
      END DO
*. Read H1(AO) 
      ZERO = 0.0D0
      CALL SETVEC(WRK(KONEEL),ZERO,NCOEF_1EL)
*
      OPEN (ITAP34,STATUS='OLD',FORM='UNFORMATTED',FILE='AOONEINT')
      REWIND ITAP34
      CALL MOLLAB('ONEHAMIL',ITAP34,6)
 2100 READ (ITAP34) (BUF(I),I=1,LBUF),(IBUF(I),I=1,LBUF),LENGTH
      DO 2200 I = 1,LENGTH
         WRK(KONEEL - 1 + IBUF(I)) = BUF(I)
 2200 CONTINUE
      IF (LENGTH .GE. 0) GO TO 2100
      CLOSE(ITAP34,STATUS='KEEP')
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' H(1el) in AO basis '
        CALL APRBLM2(WRK(KONEEL),NDEG,NDEG,NST,1)
      END IF
C  APRBLM2(A,LROW,LCOL,NBLK,ISYM)
*. Transform
C     TRAN_SYM_BLOC_MAT3(AIN,X,NBLOCK,LX_ROW,LX_COL,AOUT,SCR,ISYM)
      CALL TRAN_SYM_BLOC_MAT3(WRK(KONEEL),WRK(KEIGVC),NST,
     &     NDEG,NEJBAS,WRK(KMOONE),WRK(KSCR1),1)
C
C     WRITE 1-E MOINTS TO DISK TEMPORARILY
C
*
      OPEN (LUONE,STATUS='UNKNOWN',FORM='UNFORMATTED',
     *      FILE='MOONEINT')
      WRITE(LUONE) NCOEF_MO_MO,(WRK(KMOONE+I-1),I=1,NCOEF_MO_MO)
      CLOSE (LUONE,STATUS='KEEP')
C
      RETURN
      END
      SUBROUTINE MOLLAB(A,LU,LUERR)
C
C  16-Jun-1986 hjaaj
C  (as SEARCH but CHARACTER*8 instead of REAL*8 labels)
C
C  Purpose:
C     Search for MOLECULE labels on file LU
C
      CHARACTER*8 A, B(4), C
      DATA C/'********'/
*
      NTEST = 0
*
    1 READ (LU,END=3,ERR=6) B
      IF(NTEST.GE.1000) WRITE(6,'(4A8)'), B(1),B(2),B(3),B(4)
      IF (B(1).NE.C) GO TO 1
      IF (B(4).NE.A) GO TO 1
      IF (LUERR.LT.0) LUERR = 0
      RETURN
C
    3 IF (LUERR.LT.0) THEN
         LUERR = -1
         RETURN
      ELSE
         WRITE(LUERR,4)A,LU
         CALL TRACE
         STOP 'ERROR (MOLLAB) MOLECULE label not found on file'
      END IF
    4 FORMAT(/' *** ERROR (MOLLAB), MOLECULE label ',A8,
     *        ' not found on unit',I4)
C
    6 IF (LUERR.LT.0) THEN
         LUERR = -2
         RETURN
      ELSE
         WRITE (LUERR,7) LU,A
         CALL TRACE
         STOP 'ERROR (MOLLAB) error reading file'
      END IF
    7 FORMAT(/' *** ERROR (MOLLAB), error reading unit',I4,
     *       /T22,'when searching for label ',A8)
      END
      SUBROUTINE READMO(XIJKL)
C
C     Written by Henrik Koch 27-Mar-1990
C
*. Modified in Aug 2005 - Memory allocated internally 
      INCLUDE 'wrkspc.inc'
      PARAMETER (LUINT = 13)
      PARAMETER (LUPRI = 6)
*. Modified for LUCIA: XIJKL added to list
C
      PARAMETER (MAXORB = 255, MAXRHF = 30, MAXVIR = 225)
      COMMON /CIPOL / NBAST, NNBAST, MORB(8), NMORBT, NORB(8),NNORB(8),
     *                NSYMHF, LBINTM, LPPOP(3,3), LSYMOP(3,3),
     *                NCMOT, NNORBX, NLAMDA(8), LUEGVC,NORBT,ISAT(128)
C
      IDUM = 0
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'READMO') 
*
      OPEN (LUINT,STATUS='UNKNOWN',FORM='UNFORMATTED',
     *      FILE='MOTWOINT')
      REWIND LUINT
C
      REWIND LUINT
      READ (LUINT)
      READ (LUINT) LBINTM, JTRLVL
C?    WRITE(6,*) 'LBINTM  :  ',LBINTM,JTRLVL
      CALL MEMMAN(KONEMO,NORBT*NORBT,'ADDL  ',2,'ONEMO ')
C     KONEMO = 1
      CALL MEMMAN(KTWOMO,LBINTM,'ADDL  ',2,'TWOMO ')
C     KTWOMO = KONEMO + NORBT*NORBT
      KBUF   = KTWOMO
      CALL MEMMAN(KIBUF,LBINTM,'ADDL  ',2,'IBUF ')
C     KIBUF  = KBUF   + LBINTM
C     KEND   = KIBUF  + LBINTM/IRAT + 1
C
C     IF ( KEND .GT. LWRK ) THEN
C        STOP 'Insufficient spaces in READMO'
C     ENDIF
C
C-----------------------------------
C     Read MO integrals into memory.
C-----------------------------------
C
      CALL REDMO1(XIJKL,WORK(KBUF),WORK(KIBUF),LBINTM,NORBT)
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'READMO') 
*
      RETURN
      END
      SUBROUTINE REDMO1(TWOMO,BUF,IBUF,LBUF,NORBT)
C
C     Written by Henrik Koch 27-Mar-1990.
C
*
*. Modified to be SIRIUS-LUCIA interface, february 1993
*
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (LUINT = 13)
      PARAMETER (LUPRI = 6)
C
C     -------------------------------------------------------
C     IRAT  = (real word length) / (integer word length)
C     IRAT2 = (real word length) / (half-integer word length)
C             if available and used, otherwise IRAT2 = IRAT
C     PARAMETER (IRAT = 2, IRAT2 = 2)
      INCLUDE 'irat.inc'
      DIMENSION TWOMO(*),
     *          BUF(LBUF),IBUF(LBUF)
      PARAMETER (MAXORB = 255, MAXRHF = 30, MAXVIR = 225)
COLD  INTEGER P,Q,R,S,A,B,C,D,E,F,G
C----- bit manipulation definitions
      PARAMETER (IBT02=3, IBT08=255, IBT10=1023, IBT16=65535)
      PARAMETER (MYSHF=16,IBTMY=IBT16,MAXLN=16)
      IBTAND(I,J) = IAND(I,J)
      IBTOR(I,J)  = IOR(I,J)
      IBTSHL(I,J) = ISHFT(I,J)
      IBTSHR(I,J) = ISHFT(I,-J)
      IBTXOR(I,J) = IEOR(I,J)
C-----
C
      REWIND LUINT
      CALL MOLLAB('MOLTWOEL',LUINT,LUERR)
C
      INDCD = 0
      NINTR = 0    
  200 READ (LUINT) BUF,IBUF,LENGTH
      IF (LENGTH .EQ. 0) GOTO 200
      IF (LENGTH .EQ. -1) GOTO 9500
      INDCDN = IBTAND(IBTSHR(IBUF(1),16),IBT16)
      IF ( INDCDN .NE. INDCD ) THEN
         INDCD = INDCDN
         IC    = IBTAND(IBTSHR(INDCD,8),IBT08)
         ID    = IBTAND(       INDCD,   IBT08)
      ENDIF
      DO 280 I = 1,LENGTH
         IA = IBTAND(IBTSHR(IBUF(I),8),IBT08)
         IB = IBTAND(       IBUF(I),   IBT08)
         NINTR = NINTR + 1
         IABCD = I2EAD(IA,IB,IC,ID)
C        write(6,*) ' IA IB IC ID IABCD ',IA,IB,IC,ID,IABCD
         TWOMO(IABCD) = BUF(I)
c
  280 CONTINUE
      GOTO 200
 9500 CONTINUE
*
      WRITE(6,*) ' Number of integrals read ', NINTR
      WRITE(6,*) ' Indeces of last integral ', IA,IB,IC,ID
      RETURN
      END
      SUBROUTINE TRACE
C
C Written 4-Dec-1983 hjaaj
C
      CALL TRACEBACK
      RETURN
      END 
      SUBROUTINE UNPCSY(SYM,UNPC,NDIM)
C
C     Written by Henrik koch
C
C     Unpack a symmetric matric of dimension NDIM
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION SYM(NDIM,NDIM),UNPC(*)
C
      ITADR(I,J) = (MAX(I,J)*(MAX(I,J) - 3))/2 + I + J
C
      DO 100 I = 1,NDIM
         DO 200 J = 1,I
            IJ = ITADR(J,I)
            SYM(I,J) = UNPC(IJ)
            SYM(J,I) = UNPC(IJ)
  200    CONTINUE
  100 CONTINUE
C
      RETURN
      END
      SUBROUTINE TRAN_SYM_BLOC_MAT3
     &(AIN,X,NBLOCK,LX_ROW,LX_COL,AOUT,SCR,ISYM)
*
* Transform a blocked matrix AIN with blocked matrix
*  X to yield blocked matrix AOUT
*
* ISYM = 1 => Input and output are     triangular packed
*      else=> Input and Output are not triangular packed
*
* Aout = X(transposed) A X
*
* Jeppe Olsen
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION AIN(*),X(*),LX_ROW(NBLOCK),LX_COL(NBLOCK)
*. Output 
      DIMENSION AOUT(*)
*. Scratch: At least twice the length of largest block 
      DIMENSION SCR(*)
*
*. To get rid of annoying and incorrect compiler warnings
      IOFFP_IN = 0
      IOFFC_IN = 0
      IOFFP_OUT = 0 
      IOFFC_OUT = 0
      IOFFX = 0
*
      DO IBLOCK = 1, NBLOCK
       IF(IBLOCK.EQ.1) THEN
         IOFFP_IN = 1
         IOFFC_IN = 1
         IOFFP_OUT= 1
         IOFFC_OUT= 1
         IOFFX = 1
       ELSE
         IOFFP_IN = IOFFP_IN + LX_ROW(IBLOCK-1)*(LX_ROW(IBLOCK-1)+1)/2
         IOFFC_IN = IOFFC_IN + LX_ROW(IBLOCK-1) ** 2
         IOFFP_OUT= IOFFP_OUT+ LX_COL(IBLOCK-1)*(LX_COL(IBLOCK-1)+1)/2
         IOFFC_OUT= IOFFC_OUT+ LX_COL(IBLOCK-1) ** 2
         IOFFX = IOFFX + LX_ROW(IBLOCK-1)*LX_COL(IBLOCK-1)
       END IF
       LXR = LX_ROW(IBLOCK)
       LXC = LX_COL(IBLOCK)
       K1 = 1
       K2 = 1 + MAX(LXR,LXC) ** 2
*. Unpack block of A
       SIGN = 1.0D0
       IF(ISYM.EQ.1) THEN
         CALL TRIPAK(SCR(K1),AIN(IOFFP_IN),2,LXR,LXR,SIGN)
C             TRIPAK(AUTPAK,APAK,IWAY,MATDIM,NDIM,SIGN)
       ELSE
         CALL COPVEC(AIN(IOFFC_IN),SCR(K1),LXR*LXR)
       END IF
*. X(T)(IBLOCK)A(IBLOCK)
       ZERO = 0.0D0
       ONE  = 1.0D0
       CALL SETVEC(SCR(K2),ZERO,LXR*LXC)
       CALL MATML7(SCR(K2),X(IOFFX),SCR(K1),LXC,LXR,LXR,LXC,LXR,LXR,
     &             ZERO,ONE,1)
C?     WRITE(6,*) ' Half transformed matrix '
C?     CALL WRTMAT(SCR(K2),LXC,LXR,LXC,LXR)

*. X(T) (IBLOCK) A(IBLOCK) X (IBLOCK)
       CALL SETVEC(SCR(K1),ZERO,LXC*LXC)
       CALL MATML7(SCR(K1),SCR(K2),X(IOFFX),LXC,LXC,LXC,LXR,LXR,LXC,
     &             ZERO,ONE,0)
C?     WRITE(6,*) ' Transformed matrix '
C?     CALL WRTMAT(SCR(K1),LXC,LXC,LXC,LXC)
*. Pack and transfer
       IF(ISYM.EQ.1) THEN
         CALL TRIPAK(SCR(K1),AOUT(IOFFP_OUT),1,LXC,LXC,SIGN)
       ELSE
         CALL COPVEC(SCR(K1),AOUT(IOFFC_OUT),LXC*LXC)
       END IF
*
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*)
        WRITE(6,*) ' output matrix TRAN_SYM_BLOC_MAT '
        WRITE(6,*) ' ==============================='
        WRITE(6,*)
        CALL APRBLM2(AOUT,LX_COL,LX_COL,NBLOCK,ISYM)      
      END IF
*
      RETURN
      END

     
      SUBROUTINE SETINT_LIPK(NPART,XE,XV,XINT1,XINT2,NINT1,NINT2)

      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'

      DIMENSION XINT1(NINT1), XINT2(NINT2)

      DO II = 1, NINT1
        XINT1(II) = 0D0
      END DO
      DO II = 1, NINT2
        XINT2(II) = 0D0
      END DO

      DO II = 1, NPART
        IABS = II
        IDX = IABS*(IABS-1)/2+IABS
        XINT1(IDX) = -0.5d0*XE
      END DO
      DO II = 1, NPART
        IABS = II + NPART
        IDX = IABS*(IABS-1)/2+IABS
        XINT1(IDX) = +0.5d0*XE
      END DO
      
      DO II = 1, NPART
        DO JJ = 1, II
          CALL PTIJKL(II+NPART,II,JJ+NPART,JJ,XV,XINT2)
          CALL PTIJKL(II,II+NPART,JJ,JJ+NPART,XV,XINT2)
        END DO
      END DO

      NTEST = 1000
      IF (NTEST.GE.100) THEN
        WRITE(6,*) 'Lipkin 1-part Hamiltonian:', NTOOB
        CALL PRSYM(XINT1,2*NPART)
      END IF

      IF (NTEST.GE.1000) THEN
        WRITE(6,*) 'Lipkin 2-part Hamiltonian:'
        DO II = 1, 2*NPART
          DO JJ = 1, 2*NPART
            DO KK = 1, 2*NPART
              DO LL = 1, 2*NPART
                XX = GTIJKL(II,JJ,KK,LL)
                IF (XX.NE.0D0)
     &               WRITE(6,'(2X,4I4,E20.10)') II,JJ,KK,LL,XX 
              END DO
            END DO
          END DO
        END DO
      END IF

      RETURN

      END
      SUBROUTINE REOSYM(NARRAY,NIRNEW,NIROLD,IRMAP,LDIRMAP)
c
c     reorder entries in NARRAY according to map IRMAP
c
c     needed when going to a subgroup relative to what
c     actually is declared in LUCIA.INP (needed during numerical
c     differentiation)
c
      IMPLICIT NONE

      INTEGER, PARAMETER ::
     &     NTEST = 00

      INTEGER, INTENT(IN) ::
     &     NIRNEW, NIROLD, LDIRMAP, IRMAP(LDIRMAP,NIRNEW)
      INTEGER, INTENT(INOUT) ::
     &     NARRAY(NIROLD)

      INTEGER ::
     &     NSCR(NIRNEW)
      INTEGER ::
     &     IDX, JDX, N

      DO IDX = 1, NIRNEW
        N = 0
        DO JDX = 1, LDIRMAP
          N = N + NARRAY(IRMAP(JDX,IDX))
        END DO
        NSCR(IDX) = N
      END DO

      IF (NTEST.GE.100) THEN
        WRITE(6,*) 'old -> new'
        WRITE(6,*) NARRAY(1:NIROLD), ' -> ',NARRAY(1:NIRNEW)
      END IF

      NARRAY(1:NIRNEW) = NSCR(1:NIRNEW)

      RETURN
      END
      SUBROUTINE NATORB3(RHO1,NSMOB,NACOBS,NINOBS,NSCOBS,
     &                  NINOB,NACOB,ISTREO,XNAT,RHO1SM,OCCNUM,
     &                  SCR,IPRDEN)
*
* Obtain natural orbitals in symmetry blocks. Input density is 
* over orbitals on GASpaces (active orbitals) in type-symmetry order
*
* Jeppe Olsen, Modification of NATORB, Sept 2005
*              Changed June 2010, so selected corresponds to all active
*              orbitals
*              
*
      IMPLICIT REAL*8(A-H,O-Z)
*. Input
      DIMENSION RHO1(NACOB,NACOB)
      INTEGER ISTREO(*)
      INTEGER NACOBS(*),NINOBS(*),NSCOBS(*)
*. Output
      DIMENSION RHO1SM(*),OCCNUM(*),XNAT(*)
*. Scratch ( Largest symmetry block )
      DIMENSION SCR(*)
*
      NTESTL = 0
      NTEST = MAX(NTESTL,IPRDEN)
*. Loop over active orbitals in output order: symmetry type
      IOBOFF = 0
      IMTOFF = 0
      IADD_ST = 0
      IADD_TS = NINOB
      DO ISMOB = 1, NSMOB
        IF(ISMOB.EQ.1) THEN
          IOBOFF     = 1
          IMTOFF     = 1
          IADD_ST    = NINOBS(1)
        ELSE
          IOBOFF     = IOBOFF + NACOBS(ISMOB-1)
          IMTOFF     = IMTOFF + NACOBS(ISMOB-1)**2
          IADD_ST    = IADD_ST + NINOBS(ISMOB) + NSCOBS(ISMOB-1)
        END IF
        LOB = NACOBS(ISMOB)
C?      WRITE(6,*) ' ISMOB, LOB, = ', ISMOB, LOB
C?      WRITE(6,*) ' IADD_TS = ', IADD_TS
*
*. Extract symmetry block of density matrix
*
*. Loop over active orbitals of symmetry ISMOB in ST order
        DO IOB = IOBOFF,IOBOFF + LOB-1
           IOB_ABS = IOB + IADD_ST
           IOB_TS = ISTREO(IOB_ABS) - IADD_TS
           IOB_REL = IOB  - IOBOFF + 1
           DO JOB = IOBOFF,IOBOFF + LOB-1
               JOB_ABS = JOB + IADD_ST
               JOB_TS = ISTREO(JOB_ABS) - IADD_TS
               JOB_REL = JOB  - IOBOFF + 1
               IF(NTEST.GE.1000) THEN
                 WRITE(6,*) ' JOB, JOB_ABS, JOB_TS, ISTREO() = ',
     &                        JOB, JOB_ABS, JOB_TS, ISTREO(JOB_ABS)
                 WRITE(6,*) ' IOB_TS, JOB_TS = ', IOB_TS, JOB_TS
                 WRITE(6,'(A,6I3)') 
     &           ' IOB_TS, JOB_TS, IOB, JOB, IOB_REL, JOB_REL  = ',
     &             IOB_TS, JOB_TS, IOB, JOB, IOB_REL, JOB_REL
               END IF
               RHO1SM(IMTOFF-1+(JOB_REL-1)*LOB+IOB_REL)
     &       = RHO1(IOB_TS,JOB_TS)
           END DO
        END DO
*
        IF(NTEST.GE.2 ) THEN
          WRITE(6,*)
          WRITE(6,*) ' Density matrix for symmetry  = ', ISMOB
          WRITE(6,*) ' ======================================='
          WRITE(6,*)
          CALL WRTMAT(RHO1SM(IMTOFF),LOB,LOB,LOB,LOB)
        END IF
*. Pack and diagonalize
        CALL TRIPAK(RHO1SM(IMTOFF),SCR,1,LOB,LOB)
        ONEM = -1.0D0
*. scale with -1 to get highest occupation numbers as first eigenvectors
        CALL SCALVE(SCR,ONEM,LOB*(LOB+1)/2)       
        CALL EIGEN(SCR,XNAT(IMTOFF),LOB,0,1)
*
        DO  I = 1, LOB   
          OCCNUM(IOBOFF-1+I) = - SCR(I*(I+1)/2) 
        END DO 
*. Order the degenerate eigenvalues so diagonal terms are maximized
        TESTY = 1.0D-11
        DO IOB = 2, LOB
          IF(ABS(OCCNUM(IOBOFF-1+IOB)-OCCNUM(IOBOFF-2+IOB))
     &       .LE.TESTY) THEN
            XII   = ABS(XNAT(IMTOFF-1+(IOB-1)  *LOB+IOB  ))
            XI1I1 = ABS(XNAT(IMTOFF-1+(IOB-1-1)*LOB+IOB-1))
            XII1  = ABS(XNAT(IMTOFF-1+(IOB-1-1)*LOB+IOB  ))
            XI1I  = ABS(XNAT(IMTOFF-1+(IOB-1)  *LOB+IOB-1))
*
            IF( XI1I.GT.XII.AND.XII1.GT.XI1I1 ) THEN
*. interchange orbital IOB and IOB -1
              CALL SWAPVE(XNAT(IMTOFF+(IOB-1)*LOB),
     &                    XNAT(IMTOFF+(IOB-1-1)*LOB),LOB)
              SS = OCCNUM(IOBOFF-1+IOB-1)
              OCCNUM(IOBOFF-1+IOB-1) = OCCNUM(IOBOFF-1+IOB)
              OCCNUM(IOBOFF-1+IOB)   = SS             
              write(6,*) ' Orbitals interchanged ',
     &        IOBOFF-1+IOB,IOBOFF-2+IOB
            END IF
          END IF
        END DO
*
        IF(NTEST.GE.1) THEN
          WRITE(6,*)
          WRITE(6,*) 
     &    ' Natural occupation numbers for symmetry = ', ISMOB
          WRITE(6,*)
     &    ' ==================================================='
          WRITE(6,*)
          CALL WRTMAT(OCCNUM(IOBOFF),1,LOB,1,LOB)
          IF(NTEST.GE.5 ) THEN
            WRITE(6,*)
            WRITE(6,*) ' Corresponding Eigenvectors(MO-MO) '
            WRITE(6,*)
            CALL WRTMAT(XNAT(IMTOFF),LOB,LOB,LOB,LOB)
          END IF
        END IF
      END DO
*. ( End of loop over orbital symmetries )
*
      RETURN
      END 
      SUBROUTINE E_SUMMARY
*
* Summarize the calculations performed and the obtained energies
*
*. Jeppe Olsen, Sicily, Sept 2009 - as I need the CAS energy for a 
*               diagonal element in ICCI..
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cfinal_e.inc'
      INCLUDE 'cgas.inc'
      CHARACTER*6 CARD
*
      WRITE(6,*) 
      WRITE(6,*) 
      WRITE(6,*) 
      WRITE(6,*) '          ***************************'
      WRITE(6,*) '          *                         *'
      WRITE(6,*) '          * Summary of calculations *'
      WRITE(6,*) '          *                         *'
      WRITE(6,*) '          ***************************'
      WRITE(6,*) 
      WRITE(6,*) 
     &' Space   Form of calc.      E(final)       Error norm Converged?'
      WRITE(6,*)
     & '---------------------------------------------------------------'
      DO JCMBSPC = 1, NCMBSPC
      DO ICALC = 1, NSEQCI(JCMBSPC)
        CARD(1:6)=CSEQCI(ICALC,JCMBSPC)(1:6)
C?      WRITE(6,'(1H , A)') 'CSEQ..', CSEQCI(ICALC,JCMBSPC)(1:8)
        IF(CONV_T(ICALC,JCMBSPC)) THEN
          WRITE(6,'(I5,8X,A,F20.12,4X,E12.5,3X,A3)')
     &          JCMBSPC,CARD,E_FINAL_T(ICALC,JCMBSPC), 
     &          ERROR_NORM_FINAL_T(ICALC,JCMBSPC), ' + '
        ELSE
          WRITE(6,'(I5,8X,A,F20.12,4X,E12.5,3X,A3)')
     &          JCMBSPC,CARD,E_FINAL_T(ICALC,JCMBSPC), 
     &          ERROR_NORM_FINAL_T(ICALC,JCMBSPC), ' - '
        END IF
      END DO
      END DO
*
      RETURN
      END
      SUBROUTINE PRINT_CMOAO(CMOAO)
*
* Print MO-AO expansion coefficient matrix CMOAO
*
*. Jeppe Olsen, June 2010 (after LUCIA has existed in about 15 years..)
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
*. Specific input
      DIMENSION CMOAO(*)
*. General input
      CHARACTER*4 AO_CENT, AO_TYPE
      COMMON/AOLABELS/AO_CENT(MXPORB),AO_TYPE(MXPORB)
*
      WRITE(6,*) ' MO - AO transformation matrix '
      WRITE(6,*) ' ------------------------------'
      NMO_PER_BLK = 10 
      IOFF_AO = 1
      IOFF_MOAO = 1
      DO ISM = 1, NSMOB
        WRITE(6,*)
        WRITE(6,*) ' MO''s with symmetry ', ISM
        WRITE(6,*)
        NMOS = NMOS_ENV(ISM)
        NAOS = NAOS_ENV(ISM)
        NBLOCK = NMOS/NMO_PER_BLK
        IF(NBLOCK*NMO_PER_BLK.LT.NMOS) NBLOCK = NBLOCK + 1
C?      WRITE(6,*) ' ISM, NMOS, NAOS, NBLOCK = ',
C?   &               ISM, NMOS, NAOS, NBLOCK
        DO IBLOCK = 1, NBLOCK
          IMO_START = (IBLOCK-1)*NMO_PER_BLK  + 1
          IMO_STOP  = MIN(NMOS,IBLOCK*NMO_PER_BLK)
          WRITE(6,'(14X, 10(2X,I4,1X))') (IMO, IMO = IMO_START,IMO_STOP)
          DO IAO = 1, NAOS
           WRITE(6,'(2X,A4,2X,A4,2X,10F7.3,(14X,10F7.3))')
     &     AO_CENT(IOFF_AO-1+IAO), AO_TYPE(IOFF_AO-1+IAO),
     &     (CMOAO(IOFF_MOAO-1+(IMO-1)*NAOS+IAO),IMO =IMO_START,IMO_STOP)
          END DO
        END DO
        IOFF_AO = IOFF_AO + NAOS
        IOFF_MOAO = IOFF_MOAO + NMOS*NAOS
      END DO
*
      RETURN
      END
      SUBROUTINE GETINT_ORIG(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,ICOUL)
*
*
* Outer routine for accessing integral block
*
* if we have unrestricted spin-orbitals (I_UNRORB.EQ.1), this is important:
* ISPCAS: 1 -- alpha alpha
* ISPCAS: 2 -- beta  beta
* ISPCAS: 3 -- alpha beta (i.e. IJ alpha, KL beta)
* ISPCAS: 4 -- beta  alpha(i.e. IJ beta,  KL alpha)
*
c      IMPLICIT REAL*8(A-H,O-Z)
*
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'cc_exc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'cgas.inc'
*
      CALL QENTER('GETIN')
      NTEST = 000
*
      IF(NTEST.GE.1) THEN
       WRITE(6,*) ' I_USE_SIMTRH in GETINT =', I_USE_SIMTRH
       WRITE(6,*) ' I_UNRORB in GETINT =', I_UNRORB
       WRITE(6,*) ' GETINT: ICC_EXC and ICOUL = ', ICC_EXC, ICOUL
       WRITE(6,*)       'ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM: ' 
       WRITE(6,'(8I4)')  ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM
      END IF
*
*. Modified July 2010: ITP = -1 now indicated all orbitals
*                          =  0 indicates inactive
*                          =  NGAS + 1 indicates secondary

      IF (ISPCAS.EQ.4) STOP 'STILL A BUG FOR ISPCAS.EQ.4!'
*
      IF(ICC_EXC.EQ.0.AND.I_USE_SIMTRH.EQ.0) THEN
*
* =======================
* Usual/Normal  integrals
* =======================
*
*. Integrals in core in internal LUCIA format
        IF(ICOUL.NE.2.AND.I_UNRORB.EQ.0) THEN
          IF (I12S.EQ.1.AND.I34S.EQ.1) THEN
            CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                    IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                    WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
          ELSE
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
          END IF
        ELSE IF (I_UNRORB.EQ.0) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.1) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.2) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2BB),
     &                  WORK(KPINT2),NSMOB,WORK(KINH1),ICOUL,0)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.3) THEN
          CALL GETINCN2(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2AB),
     &                  WORK(KPINT2AB),NSMOB,WORK(KINH1),ICOUL,1)
        ELSE IF (I_UNRORB.EQ.1.AND.ISPCAS.EQ.4) THEN
          CALL GETINCN2(XINT,KTP,KSM,LTP,LSM,ITP,ISM,JTP,JSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2AB),
     &                  WORK(KPINT2AB),NSMOB,WORK(KINH1),ICOUL,1)
        ELSE
          WRITE(6,*) 'WRONG ISPCAS IN GETINT (',ISPCAS,')'
          STOP 'getint'
        END IF
      ELSE IF (ICC_EXC.EQ.1.AND.I_USE_SIMTRH.EQ.0) THEN
*
* ============================
* Coupled Cluster coefficients 
* ============================
* 
        IF(ICOUL.EQ.1) THEN
          IKLJ = 0 
          IJ_TRNSP = 1
        ELSE
          IKLJ = 1
          IJ_TRNSP = 0
        END IF
*. IJ_TRNSP: RSBB2BN requires blocks for e(ijkl) in the form C(ji,kl)
*. Amplitudes fetched from KCC1, KCC2 used as scratch 
        CALL GET_DX_BLK(ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,WORK(KCC1+NSXE),
     &                  XINT,1,IXCHNG,IKLJ,IKSM,JLSM,WORK(KCC2),
     &                  IJ_TRNSP )
C            GET_DX_BLK(IGAS,ISM,JGAS,JSM,KGAS,KSM,LGAS,LSM, 
C    &                  C,CBLK,IEXP,IXCHNG,IKLJ,IKSM,JLSM,SCR)
      ELSE IF( I_USE_SIMTRH.EQ.1) THEN
*. Use similarity transformed integrals
        IF(I_UNRORB.EQ.0) THEN
C          IF(ICOUL.NE.2) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0)
C          ELSE
C            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
C     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH),
C     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
C     &                  ICOUL,0)
C          END IF

        ELSE
          IF(ISPCAS.EQ.1) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AA),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0)
          ELSE IF(ISPCAS.EQ.2) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                  IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_BB),
     &                  WORK(KPINT2_SIMTRH),NSMOB,WORK(KINH1_NOCCSYM),
     &                  ICOUL,0)
          ELSE IF(ISPCAS.EQ.3) THEN
            CALL GETINCN2_NOCCSYM(XINT,ITP,ISM,JTP,JSM,KTP,KSM,LTP,LSM,
     &                 IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AB),
     &                 WORK(KPINT2_SIMTRH_AB),NSMOB,WORK(KINH1_NOCCSYM),
     &                 ICOUL,1)
          ELSE IF(ISPCAS.EQ.4) THEN
            CALL GETINCN2_NOCCSYM(XINT,KTP,KSM,LTP,LSM,ITP,ISM,JTP,JSM,
     &                 IXCHNG,IKSM,JLSM,WORK(KINT2_SIMTRH_AB),
     &                 WORK(KPINT2_SIMTRH_AB),NSMOB,WORK(KINH1_NOCCSYM),
     &                 ICOUL,1)
          ELSE
            WRITE(6,*) 'WRONG ISPCAS IN GETINT (',ISPCAS,')'
            STOP 'getint'
          END IF
        END IF

      END IF
*
      IF(NTEST.GE.100) THEN
        NI = NOBPTS_GN(ITP,ISM)
        NK = NOBPTS_GN(KTP,KSM)
*
        IF(IKSM.EQ.0) THEN
          NIK = NI * NK
        ELSE
          NIK = NI*(NI+1)/2
        END IF
*
        NJ = NOBPTS_GN(JTP,JSM)
        NL = NOBPTS_GN(LTP,LSM)
*
        IF(JLSM.EQ.0) THEN
          NJL = NJ * NL
        ELSE
          NJL = NJ*(NJ+1)/2
        END IF
        WRITE(6,*) ' 2 electron integral block for TS blocks '
        WRITE(6,*) ' Icoul :', ICOUL
        WRITE(6,*) ' Ixchng:', IXCHNG
        WRITE(6,*) ' ISPCAS:', ISPCAS
        WRITE(6,*) ' Integrals from GETINT:'
        IF(ICOUL.EQ.0) THEN
          WRITE(6,'(1H ,4(A,I2,A,I2,A))')
     &    '(',ITP,',',ISM,')','(',KTP,',',KSM,')',
     &    '(',JTP,',',JSM,')','(',LTP,',',LSM,')'
          CALL WRTMAT(XINT,NIK,NJL,NIK,NJL)
        ELSE
          WRITE(6,'(1H ,4(A,I2,A,I2,A))')
     &    '(',ITP,',',ISM,')','(',JTP,',',JSM,')',
     &    '(',KTP,',',KSM,')','(',LTP,',',LSM,')'
          CALL WRTMAT(XINT,NI*NJ,NK*NL,NI*NJ,NK*NL)
        END IF
      END IF
*
      CALL QEXIT('GETIN')
C     STOP ' Jeppe forced me to stop in GETINT '
      RETURN
      END
      SUBROUTINE EXT_CP_AC_GASBLKS(NSMOB,NGAS,NOBPTS_GN,MXPNGAS,
     &           IEORC,NTOOBS, NACOBS,AACT,AALL)
*
* Two matrices AACT and AALL are given. AACT is over active
* gas blocks whereas AALL is over all orbitals.
*
* IEORC = 1: Extract active blocks from AALL and place in AACT
* IEORC = 2: Copy active blocks form AACT to AALL
*
* Jeppe Olsen, Feb. 2011
*
      INCLUDE 'implicit.inc'
      DIMENSION  AACT(*), AALL(*)
      INTEGER NOBPTS_GN(0:MXPNGAS,*), NACOBS(NSMOB),NTOOBS(NSMOB)
*
      NTEST = 000
      IF(NTEST.GE.100) THEN 
        WRITE(6,*) 'EXT_CP_AC_GASBLKS speaking'
        WRITE(6,*) '==========================='
        WRITE(6,*)
        WRITE(6,*) ' IEORC = ', IEORC
      END IF
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' NACOBS = ' 
        CALL IWRTMA(NACOBS,1,NSMOB,1,NSMOB)
        WRITE(6,*) ' Input active matrix '
        CALL APRBLM2(AACT,NACOBS,NACOBS,NSMOB,0)
        WRITE(6,*) ' Input matrix over all orbitals '
        CALL APRBLM2(AALL,NTOOBS,NTOOBS,NSMOB,0)
      END IF
     
*
      DO ISM = 1, NSMOB
        IF(ISM.EQ.1) THEN  
          IOFF_ALL = 1
          IOFF_ACT = 1
        ELSE
          IOFF_ALL = IOFF_ALL + NTOOBS(ISM-1)**2
          IOFF_ACT = IOFF_ACT + NACOBS(ISM-1)**2
        END IF
        IF(NTEST.GE.10000) WRITE(6,*) ' ISM, IOFF_ALL, IOFF_ACT = ',
     &  ISM, IOFF_ALL, IOFF_ACT
        IOFFI_ALL = 1
        IOFFI_ACT = 1
        NOB_ALL = NTOOBS(ISM)
        NOB_ACT = NACOBS(ISM)
        IF(NTEST.GE.10000) WRITE(6,*) ' NOB_ALL, NOB_ACT = ', 
     &                                 NOB_ALL, NOB_ACT
        DO ISPC = 0, NGAS +1
         IF(ISPC.NE.0) IOFFI_ALL = IOFFI_ALL + NOBPTS_GN(ISPC-1,ISM)
         IF(ISPC.GT.1) IOFFI_ACT = IOFFI_ACT + NOBPTS_GN(ISPC-1,ISM)
         NI = NOBPTS_GN(ISPC,ISM)
         IOFFJ_ALL = 1
         IOFFJ_ACT = 1
         DO JSPC = 0, NGAS + 1
           NJ = NOBPTS_GN(JSPC,ISM)
           IF(JSPC.NE.0) IOFFJ_ALL = IOFFJ_ALL + NOBPTS_GN(JSPC-1,ISM)
           IF(JSPC.GT.1) IOFFJ_ACT = IOFFJ_ACT + NOBPTS_GN(JSPC-1,ISM)
           IF((0.LT.ISPC.AND.ISPC.LE.NGAS).AND.
     &        (0.LT.JSPC.AND.JSPC.LE.NGAS)) THEN
              IF(NTEST.GE.10000) WRITE(6,*) ' ISPC, JSPC = ', ISPC, JSPC
              IF(NTEST.GE.10000) WRITE(6,*) ' NI, NJ = ', NI, NJ         
              DO I = 1, NI
              DO J = 1, NJ
               IF(IEORC.EQ.1) THEN
                 AACT(IOFF_ACT+ (J+IOFFJ_ACT-2)*NOB_ACT+ I+IOFFI_ACT-2)
     &          =AALL(IOFF_ALL+ (j+IOFFJ_ALL-2)*NOB_ALL+ I+IOFFI_ALL-2)
               ELSE
                 AALL(IOFF_ALL+ (j+IOFFJ_ALL-2)*NOB_ALL+ I+IOFFI_ALL-2)
     &          =AACT(IOFF_ACT+ (J+IOFFJ_ACT-2)*NOB_ACT+ I+IOFFI_ACT-2)
*
                IF(NTEST.GE.10000) THEN
                  WRITE(6,'(A,6(1X,I3))') 
     &            ' I, J, I_ALL, J_ALL, I_ACT, J_ACT = ',
     &              I, J, I+IOFFI_ALL-1,j+IOFFJ_ALL-1,
     &              I+IOFFI_ACT-1, J+IOFFJ_ACT-1
                  WRITE(6,'(A,2(1X,I6))') ' IJ_ACT, IJ_ALL = ',
     &            IOFF_ACT+ (J+IOFFJ_ACT-2)*NOB_ACT+ I+IOFFI_ACT-2,
     &            IOFF_ALL+ (j+IOFFJ_ALL-2)*NOB_ALL+ I+IOFFI_ALL-2
                END IF
*
               END IF
             END DO
             END DO
*            ^ End of loop over I,J
           END IF
*          ^ End if I,J are active orbitals
         END DO
       END DO
*.     ^ End of loop over ISPC, JSPC
      END DO
*.    ^ End of loop over ISM
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Final matrix  over active orbitals'
        CALL APRBLM2(AACT,NACOBS,NACOBS,NSMOB,0)
        WRITE(6,*) ' Final matrix over all orbitals'
        CALL APRBLM2(AALL,NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      RETURN
      END
      SUBROUTINE SIGDEN_CI_ORIG(CB,HCB,LUC,LUHC,C,HC,ISIGDEN)
*
* Outer routine for common sigma vector generation(ISIGDEN=1) or 
* density matrix construction(ISIGDEN=2)
* GAS version 
*
* Jeppe Olsen, April 2011, from MV7
*
c      IMPLICIT REAL*8(A-H,O-Z)
c      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc.inc'
*
* =====
*.Input
* =====
*
*.Definition of c and sigma
      INCLUDE 'cands.inc'
*
*./ORBINP/: NACOB used
      INCLUDE 'orbinp.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cecore.inc'
      COMMON/CMXCJ/MXCJ,MAXK1_MX,LSCMAX_MX
*. Two blocks of C or Sigme
      DIMENSION CB(*),HCB(*)
*. Two vectors of C or Sigma (for ICISTR = 1)
      DIMENSION C(*),HC(*)
*
      CALL QENTER('SIDEC')
      IDUM = 0
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'SIDEC ')
*
      MAXK1_MX = 0
      LSCMAX_MX = 0
      IF(ISSPC.LE.NCMBSPC) THEN
        IATP = 1
        IBTP = 2
      ELSE
        IATP = IALTP_FOR_GAS(ISSPC)
        IBTP = IBETP_FOR_GAS(ISSPC)
      END IF
*
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
*. Offset for supergroups 
      IOCTPA = IBSPGPFTP(IATP)
      IOCTPB = IBSPGPFTP(IBTP)
*
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
*. Arrays giving allowed type combinations 
      CALL MEMMAN(KSIOIO,NOCTPA*NOCTPB,'ADDL  ',2,'SIOIO ')
      CALL IAIBCM(ISSPC,dbl_mb(KSIOIO))
*. Arrays for additional symmetry operation
      IF(IDC.EQ.3.OR.IDC.EQ.4) THEN
        CALL MEMMAN(KSVST,NSMST,'ADDL  ',2,'SVST  ')
        CALL SIGVST(WORK(KSVST),NSMST)
      ELSE
         KSVST = 1
      END IF
*. Arrays giving block type
      CALL MEMMAN(KSBLTP,NSMST,'ADDL  ',1,'SBLTP ')
      CALL ZBLTP(ISMOST(1,ISSM),NSMST,IDC,int_mb(KSBLTP),WORK(KSVST))
*. Arrays for partitioning of sigma  
      NTTS = MXNTTS
      CALL MEMMAN(KLSLBT ,NTTS  ,'ADDL  ',1,'CLBT  ')
      CALL MEMMAN(KLSLEBT ,NTTS  ,'ADDL  ',1,'CLEBT ')
      CALL MEMMAN(KLSI1BT,NTTS  ,'ADDL  ',1,'CI1BT ')
      CALL MEMMAN(KLSIBT ,8*NTTS,'ADDL  ',1,'CIBT  ')
*. Batches  of C vector
      IF (ISIMSYM.EQ.0) THEN
        LBLOCK = MXSOOB
      ELSE
        LBLOCK = MXSOOB_AS
      END IF
      LBLOCK = MAX(LBLOCK,LCSBLK)
C     WRITE(6,*) ' ECORE in MV7 =', ECORE
      CALL PART_CIV2(IDC,int_mb(KSBLTP),int_mb(KNSTSO(IATP)),
     &     int_mb(KNSTSO(IBTP)),NOCTPA,NOCTPB,NSMST,LBLOCK,
     &     dbl_mb(KSIOIO),ISMOST(1,ISSM),
     &     NBATCH,int_mb(KLSLBT),int_mb(KLSLEBT),
     &     int_mb(KLSI1BT),int_mb(KLSIBT),0,ISIMSYM)
*. Number of BLOCKS
        NBLOCK = IFRMR(int_mb(KLSI1BT),1,NBATCH)
     &         + IFRMR(int_mb(KLSLBT),1,NBATCH) - 1
C?      WRITE(6,*) ' Number of blocks ', NBLOCK

      IF(I12.EQ.2) THEN
        IDOH2 = 1
      ELSE
        IDOH2 = 0
      END IF
*
      IF(ICISTR.EQ.1) THEN
       LLUC = 0
       LLUHC = 0
      ELSE 
       LLUC = LUC
       LLUHC = LUHC
      END IF
*
        CALL SIGDEN2_CI(CB,HCB,NBATCH,WORK(KLSLBT),WORK(KLSLEBT),
     &       WORK(KLSI1BT),WORK(KLSIBT),LLUC,LLUHC,C,HC,ECORE,ISIGDEN)
*. Eliminate local memory
      CALL MEMMAN(KDUM ,IDUM,'FLUSM ',2,'SIDEC ')
*
      CALL QEXIT('SIDEC')
*
      RETURN
      END
      SUBROUTINE SIGDEN2_CI(CB,SB,NBATS,LBATS,LEBATS,I1BATS,IBATS,
     &           LUC,LUHC,CV,SV,ECORE,ISIGDEN)
*
* Common routine for Sigma vector/density matrix construction
*
*
* Jeppe Olsen   April 2011, form RASSG3
*
* =====
* Input
* =====
*

      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cprnt.inc'
*. Batches of sigma
      INTEGER LBATS(*),LEBATS(*),I1BATS(*),IBATS(8,*)
*.Scratch
      DIMENSION SB(*),CB(*)
*. Input/output if ICISTR = 1
      DIMENSION SV(*),CV(*)
*
      CALL QENTER('SIDE2')
      NTEST = 0
      NTEST = MAX(NTEST,IPRCIX)
      IF(NTEST.GE.20) THEN
        WRITE(6,*) ' ======================'
        WRITE(6,*) ' SIGDEN2_CI speaking:'
        WRITE(6,*) ' ======================'
        WRITE(6,*) ' NBATS = ',NBATS
      END IF
*
      IF(LUHC.GT.0) CALL REWINO(LUHC)
* Loop over batches over sigma blocks
      IOFF_S = 1
      DO JBATS = 1, NBATS
*. Read current batch of sigma (left hand vector) if density is constructed
        IF(ISIGDEN.EQ.2) THEN
*. Transfer S block from  disc
         DO ISBLK = I1BATS(JBATS),I1BATS(JBATS)+ LBATS(JBATS)-1
          IATP = IBATS(1,ISBLK)
          IBTP = IBATS(2,ISBLK)
          IASM = IBATS(3,ISBLK)
          IBSM = IBATS(4,ISBLK)
          IOFF = IBATS(6,ISBLK)
          LEN  = IBATS(8,ISBLK)
C?        write(6,*) 'SIGDEN2_CI: IOFF, SB(IOFF)',IOFF,SB(IOFF)
          IF(ICISTR.NE.1) THEN
            CALL IFRMDS(LEN2,1,-1,LUHC)
            CALL FRMDSCN(SB(IOFF),LEN,-1,LUHC)
          ELSE
            CALL COPVEC(SV(IOFF_S),SB(IOFF),LEN)
            IOFF_S = IOFF_S + LEN
          END IF
         END DO
        END IF! End if Sigma generation
*. Obtain sigma or density for batch of sigma blocks
        CALL SIGDEN3_CI(LBATS(JBATS),IBATS(1,I1BATS(JBATS)),1,
     &       CB,SB,LUC,0,0,0,0,0,CV,ECORE,ISIGDEN)
*
        IF(ISIGDEN.EQ.1) THEN
*. Transfer S block to permanent storage
         DO ISBLK = I1BATS(JBATS),I1BATS(JBATS)+ LBATS(JBATS)-1
          IATP = IBATS(1,ISBLK)
          IBTP = IBATS(2,ISBLK)
          IASM = IBATS(3,ISBLK)
          IBSM = IBATS(4,ISBLK)
          IOFF = IBATS(6,ISBLK)
          LEN  = IBATS(8,ISBLK)
C?        write(6,*) 'SIGDEN2_CI: IOFF, SB(IOFF)',IOFF,SB(IOFF)
          IF(ICISTR.NE.1) THEN
            CALL ITODS(LEN,1,-1,LUHC)
            CALL TODSC(SB(IOFF),LEN,-1,LUHC)
          ELSE
            CALL COPVEC(SB(IOFF),SV(IOFF_S),LEN)
            IOFF_S = IOFF_S + LEN
          END IF
         END DO
        END IF! End if Sigma generation
*
       END DO
*
      IF(ICISTR.NE.1) CALL ITODS(-1,1,-1,LUHC)
      IF(NTEST.GE.100) THEN
        IF(ICISTR.NE.1) THEN
          WRITE(6,*) ' Final S-vector on disc'
          CALL WRTVCD(SB,LUHC,1,-1)
        ELSE
          LEN_S = IOFF_S - 1
          WRITE(6,*) ' Final S-vector'
          CALL WRTMAT(SV,1,LEN_S,1,LEN_S)
        END IF
      END IF
      IF(NTEST.GE.100) WRITE(6,*) ' Leaving SIGDEN2_CI'
*
      CALL QEXIT('SIDE2')
      RETURN
      END
      SUBROUTINE Z_ACT_INTLISTS
*
* Define active lists of two-electron integrals including pointers
* to integrals and block offsets./
* The possible integral lists must have been defined in 
* Z_TYP_2INTLISTS.
*
* The pointers to the integral arrays are first defined,
* but are then negated to indicate that they do not
* contain integrals
* 
*
*. Jeppe Olsen, May 2011
*. Last modification; May 22, 2013; Jeppe Olsen; IE2LIST_1G_BIO added 
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cintfo.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'lucinp.inc'
*. Local scratch
      INTEGER I2ELIST_INUSE(MXP2EIARR),IOCOBTP_INUSE(MXP2EIARR)
*
      NTEST = 100
*. Integrals over all active orbitals must always be included
*. List with zero free indices
      I2ELIST_INUSE(1) = 1
      IOCOBTP_INUSE(1) = 1
      N2EI_LIST_INUSE = 1
*
      IF(I_DO_MCSCF.EQ.1) THEN
*. MCSCF calculation will be performed
*. Integrals for gradient calculation
       N2EI_LIST_INUSE = 2
       I2ELIST_INUSE(N2EI_LIST_INUSE) = 2
       IOCOBTP_INUSE(N2EI_LIST_INUSE) = 1
*. At the moment, also full Hessian transformation is set up
       N2EI_LIST_INUSE = N2EI_LIST_INUSE + 1
       I2ELIST_INUSE(N2EI_LIST_INUSE) = 3
       IOCOBTP_INUSE(N2EI_LIST_INUSE) = 2
      END IF
*. Allocate also complete transformation- for simplicity...
      I_ALLO_ALSO_FULL_INTLIST = 1
      IF(I_ALLO_ALSO_FULL_INTLIST.EQ.1) THEN
        IF(NTEST.GE.1000) 
     &  WRITE(6,*) ' Array allocated for complete transformed int.list'
        N2EI_LIST_INUSE = N2EI_LIST_INUSE + 1
        I2ELIST_INUSE(N2EI_LIST_INUSE) = 5
        IOCOBTP_INUSE(N2EI_LIST_INUSE) = 2
      END IF
*
      IF(I_DO_NORTCI.EQ.1) THEN
*. Bioorthogonal complete integral list
        N2EI_LIST_INUSE = N2EI_LIST_INUSE + 1
        I2ELIST_INUSE(N2EI_LIST_INUSE) = IE2LIST_FULL_BIO
        IOCOBTP_INUSE(N2EI_LIST_INUSE) = 2
*. Bioorthogonal integral list with one free index
        N2EI_LIST_INUSE = N2EI_LIST_INUSE + 1
        I2ELIST_INUSE(N2EI_LIST_INUSE) = IE2LIST_1G_BIO
C       IOCOBTP_INUSE(N2EI_LIST_INUSE) = 2
        IOCOBTP_INUSE(N2EI_LIST_INUSE) = 1
      END IF
*  
      IZERO = 0
      CALL ISETVC(KPINT2_A,IZERO,MXP2EIARR)
      CALL ISETVC(KPLSM2_A,IZERO,MXP2EIARR)
      CALL ISETVC(KINT2_A,IZERO,MXP2EIARR)
*. And allocate space
      NBINT2 = NSMOB**3
C  I2ELIST_INUSE(MXPI2ARR),IOCOBTP_INUSE(MXPI2ARR)
      INTSM_A = 1
      DO IILIST = 1, N2EI_LIST_INUSE
        ILIST = I2ELIST_INUSE(IILIST)
        IOCTP_L = IOCOBTP_INUSE(IILIST)
        N_L = IE2LIST_N(ILIST)
        IB_L = IE2LIST_IB(ILIST)
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' IILIST, ILIST, N_L = ', IILIST,ILIST,N_L
          WRITE(6,*) ' IOCTP_L = ', IOCTP_L
        END IF
        DO IARR = IB_L, IB_L-1+N_L
          IIARR = IE2LIST_I(IARR)
          IOCOBTP_G(IIARR) = IOCTP_L
          IF(NTEST.GE.1000)
     &    WRITE(6,*) ' ILIST, IARR, IIARR = ', ILIST, IARR, IIARR
          CALL MEMMAN(KPINT2_A(IIARR),NBINT2,'ADDL  ',2,'PINT2A')
          CALL MEMMAN(KPLSM2_A(IIARR),NBINT2,'ADDL  ',2,'LSM2A ')
          LINT = N2INTARR_G(INTSM_A,IIARR,IOCTP_L)
          NINT2_G(IIARR) = LINT
          INTSM_G(IIARR) = INTSM_A
C?        WRITE(6,*) ' JEPTEST, LINT = ', LINT
          CALL MEMMAN(KINT2_A(IIARR),LINT,'ADDL  ',2,'INT2A ')
       END DO
      END DO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Active arrays of two-electron integrals '
        WRITE(6,*) ' ======================================= '
        WRITE(6,*)
        WRITE(6,*) ' Type   Length   Sym  Ocobtp  Pointer'
        WRITE(6,*) ' ====================================='
        DO IILIST = 1, N2EI_LIST_INUSE
          ILIST = I2ELIST_INUSE(IILIST)
          IOCTP_L = IOCOBTP_INUSE(IILIST)
          N_L = IE2LIST_N(ILIST)
          IB_L = IE2LIST_IB(ILIST)
C         WRITE(6,*) ' IILIST, ILIST, N_L = ', IILIST,ILIST,N_L
C         WRITE(6,*) ' IOCTP_L = ', IOCTP_L
          DO IARR = IB_L, IB_L-1+N_L
            IIARR = IE2LIST_I(IARR)
            INTSM_L = INTSM_G(IIARR)
            IOCOBTP_L = IOCOBTP_G(IIARR)
            NINT2_L = NINT2_G(IIARR)
            KINT2_L = KINT2_A(IIARR)
            WRITE(6,'(3X,I2,2X,I8,4X,I2,4X,I1,2X,I9)')
     &      IIARR, NINT2_L, INTSM_L, IOCOBTP_L, KINT2_L
         END DO
        END DO
COLD    WRITE(6,*) '(Pointers have been negated )'
      END IF
*
      RETURN
      END
      FUNCTION ISTRNM2(IOCC,NORB,NEL,Z,NEWORD,IOFFSETS,IREORD,IRELNUM)
*
* Adress of string IOCC
*
* IF IREORD = 1, reordered address is used
* IF IRELNUM =1, address relative to start of symmetry block is used
*
* Jeppe Olsen, obtained from ISTRNM in June 2011
*
*. General input
      INTEGER NEWORD(*),Z(NORB,*),IOFFSETS(*)
*. Specific input
      INTEGER IOCC(*) 
*. Reverse lexical number
      IZ = 1
      DO 100 I = 1,NEL
        IZ = IZ + Z(IOCC(I),I)
  100 CONTINUE
*. Reordered
      IF(IREORD.EQ.0) THEN
        ISTRNM = IZ
      ELSE
        ISTRNM = NEWORD(IZ)
      END IF
*. and relative to start of symmetry block
      IF(IRELNUM.EQ.1) THEN
*. Symmetry of string
        ISYM = ISYMST(IOCC,NEL)
        ISTRNM = ISTRNM - IOFFSETS(ISYM) + 1
      END IF
      ISTRNM2 = ISTRNM
*
      NTEST = 0
      IF ( NTEST .GT. 1 ) THEN
        WRITE(6,*) ' String'
        CALL IWRTMA(IOCC,1,NEL,1,NEL)
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' Z matrix '
          CALL IWRTMA(Z,NORB,NEL,NORB,NEL)
        END IF
        WRITE(6,'(A,2I6)') 
     &  ' ISYM and OFFSET: ', ISYM, IOFFSETS(ISYM)
        WRITE(6,'(A,2I6)') 
     &  ' Lexical and final address of string ',IZ,ISTRNM2
      END IF
*
      RETURN
      END
      SUBROUTINE MOINF_FRAG
*
* Obtain information about fragments of molecule
* for the VB project and for fragment orbitals for other cases
*
*. Jeppe Olsen, July 2011
*               Modified May 2012
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'clunit.inc'
      INCLUDE 'fragmol.inc'
*
      NTEST = 100
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MOINF_FRAG speaking '
        WRITE(6,*) '====================='
        WRITE(6,'(A,A)') ' Environment ', ENVIRO
        WRITE(6,'(A,I1)') ' I_USE_LUCIA_FRAGMENTS = ', 
     &                      I_USE_LUCIA_FRAGMENTS
      END IF
*
      IF(ENVIRO(1:6).EQ.'DALTON') THEN
*. Read info from Sirius interface file 
        DO IFRAG = 0, NFRAG_TP
          IF(IFRAG.EQ.0.OR.I_USE_LUCIA_FRAGMENTS.EQ.0) THEN
            CALL MOINF_GN_DALTON(IFRAG,CSIRIFC(IFRAG))
          ELSE
            CALL MOINF_GN_LUCIA(IFRAG,CLUCINF(IFRAG))
          ENDIF
        END DO
      ELSE IF(ENVIRO(1:5).EQ.'LUCIA') THEN
          CALL MOINF_GN_LUCIA(IFRAG,CLUCINF(IFRAG))
      ELSE
        WRITE(6,*) ' MOINF_GN called with unprogrammed environment ', 
     &               ENVIRO
        STOP       ' MOINF_GN called with unprogrammed environment ' 
      END  IF
*
      RETURN
      END
      SUBROUTINE MOINF_GN_DALTON(IFRAG,INFSIRFIL)
*
* READ in information about fraction IFRAG from  DALTON
* interface file INFSIRFIL. Save in in arrays for fragment IFRAG
* CMOAO_FRAG matrix is read to KKCMOAO_FRAG(IFRAG) which is also allocated 
* here
* 
* If IFRAG = 0, then file SIRIFC is opened
*
* INFSIRFIL is pt a 8 character string with name SIRIFC_X, X = 1, 9
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc-static.inc'
      INCLUDE 'irat.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'fragmol.inc'
      INCLUDE 'lucinp.inc'
      CHARACTER*8 INFSIRFIL
*. Scratch
      INTEGER ISCR(MXPOBS)
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' MOINF_GN_DALTON called for IFRAG = ', IFRAG
      END IF
*
      LUINFSIR = 16
      IF(IFRAG.EQ.0) THEN
       OPEN(LUINFSIR,STATUS='OLD',FORM='UNFORMATTED',FILE='SIRIFC')
      ELSE
       OPEN(LUINFSIR,STATUS='OLD',FORM='UNFORMATTED',FILE=INFSIRFIL)
      END IF
*
      REWIND LUINFSIR
      CALL MOLLAB('TRCCINT ',LUINFSIR,6)
*
      ZERO = 0.0D0
      CALL SETVEC(NBAS_FRAG(1,IFRAG),IZERO,MXPOBS)
      READ (LUINFSIR) NSYMF,NORBF,NBASF,NCMOF,(ISCR(I),I=1,NSYMF),
     *              (ISCR(I),I=1,NSYMF),(NBAS_FRAG(I,IFRAG),I=1,NSYMF),
     *              POTNUCF,EMCSCFF
*. NCMO is number of coefficients in MO-AO transformation. 
* May also be calculated from NBAS_FRAG. Check
C                  NDIM_1EL_MAT(IHSM,NRPSM,NCPSM,NSM,IPACK)
      NDIM_CFRAG = NDIM_1EL_MAT(1,NBAS_FRAG(1,IFRAG),NBAS_FRAG(1,IFRAG),
     &             NSYMF,0)
      IF(NDIM_CFRAG.NE.NCMOF) THEN 
        WRITE(6,*) 
     &  ' Difference between computed and read dimension of CMO_FRAG '
        WRITE(6,*) ' NDIM_CFRAG, NCMOF = ', NDIM_CFRAG,NCMOF
        WRITE(6,*) ' NSYMF = ', NSYMF
        STOP       
     &  ' Difference between computed and read dimension of CMO_FRAG '
      END IF
      IF(NTEST.GE.100) WRITE(6,*) ' Number MO-AO coefs' , NDIM_CFRAG
*. Skip a record of orbital energies and symmetries
      READ(LUINFSIR)
*. Allocate memory and read MO-AO transformation matrix
      CALL MEMMAN(KCMOAO_FRAG(IFRAG),NDIM_CFRAG,'ADDL  ',2,'MOAO_F')
      READ(LUINFSIR) (WORK(KCMOAO_FRAG(IFRAG)-1+I),I=1, NDIM_CFRAG)
      CLOSE(LUINFSIR,STATUS='KEEP')
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MO- AO transformation matrix for fragment', IFRAG  
        WRITE(6,*) ' ================================================'
        CALL APRBLM_F7(WORK(KCMOAO_FRAG(IFRAG)),
     &       NBAS_FRAG(1,IFRAG),NBAS_FRAG(1,IFRAG),NSYMF,0)
      END IF
*
      RETURN
      END
      SUBROUTINE ASSEMBLE_MO_FROM_FRAGMENTS_DEFINE
*
* A set of fragment MO's have been defined to LUCIA
* obtain set of MO's by combining the MO's of the fragments
*
* It is assumed and required that the fragments are given in the 
* same order as in the DALTON calculation on the whole molecule
*
* Read in specification on how MO's should be obtained from the 
* MO's on the fragments
*
* Fragment 0 is the complete molecule, but use of this is pt not 
* enabled...
*
* Jeppe Olsen, July 2011 for the VB project
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'fragmol.inc'
      INCLUDE 'cgas.inc'
*
      CHARACTER*102 CARD
      CHARACTER*102 CARD1
      CHARACTER*102 LASTCARD
      CHARACTER*102 CARDX
*
*. Local  scratch for decoding multi-item lines, atmost 32 items per line
      PARAMETER(MX_ITEM = 42)
      CHARACTER*102 ITEM(MX_ITEM),  ITEMX 
      INTEGER INT_ITEM(MX_ITEM)
*
*. Read in 
*
*. Loop over fragments for which orbitals should be constructed.
*. Note that if the same atoms, say Cr is included more than once,
*. as in for example Cr2, then the same fragment is specified twice
*. to get orbitals for both atoms 
*. The sum of the NFRAG_IN fragments should be the whole molecule
*
* If one is doing a study of the reaction H2O + H2O and uses
* H2O as the two fragments, the MOLECULE calculation should 
* thus have been set up with the ATOMS: H H O H H O
      LUIN = 5
*. First orbital and number of  orbitals of  given sym and fragment that should 
*. be included as MO's in IGAS for given sym
      IREAD_NEW_OR_OLD = 1
      NERROR = 0
      IF(IREAD_NEW_OR_OLD.EQ.1) THEN
*. Number of symmetries for this molecule
        DO IFRAG = 1, NFRAG_MOL
         READ(LUIN,*) NSMOB_FRAG(IFRAG) 
         NSMOB_L = NSMOB_FRAG(IFRAG)
         DO IGAS = 0, NGAS + 1
           READ(LUIN,'(A)') CARD1
           CALL LFTPOS(CARD1,MXPLNC)
           CALL UPPCAS(CARD1,MXPLNC)
*. A line can be one of the following:
* NONE: no orbitals in this space
* NIRREP numbers giving dim of each irrep for this space
           CALL DECODE2_LINE(CARD1,MXPLNC,NITEM,ITEM,MX_ITEM)
           ITEMX = ITEM(1)
           IF(ITEMX(1:4).EQ.'NONE') THEN
             DO ISYM = 1, NSMOB_L
              N_GS_SM_BAS_FRAG(IGAS,ISYM,IFRAG) = 0
             END DO
           ELSE 
*. I expect that NIRREP integers are given
             IF(NITEM.NE.NSMOB_L) THEN
              WRITE(6,*) ' Erroneous input in ASSEMBLE..: '
              WRITE(6,'(72A)') CARD1
              WRITE(6,*) ' Specify either:   NONE '
              WRITE(6,*) ' Or NIRREP integers seperated by comma '
              NERROR = NERROR + 1
             END IF
*. Well assume NIRREP integers
             DO IRREP = 1, NSMOB_L
              CALL CHAR_TO_INTEGER(ITEM(IRREP),
     &             N_GS_SM_BAS_FRAG(IGAS,IRREP,IFRAG),
     &             MXPLNC)
             END DO
           END IF
         END DO ! Loop over IGAS
        END DO ! loop over fragment
      ELSE ! Switch between new and old input
        DO IFRAG = 1, NFRAG_MOL
         DO ISYM = 1, NSMOB
          READ(5,*) 
     &    (N_GS_SM_BAS_FRAG(IGAS,ISYM,IFRAG), IGAS = 0,NGAS + 1)
         END DO
C?       WRITE(6,*)  'N_GS_SM_BAS_FRAG read in '
C?       WRITE(6,*) 
C?   &   (N_GS_SM_BAS_FRAG(IGAS,ISYM,IFRAG), IGAS = 0,NGAS + 1)
        END DO
      END IF
*
      NTEST = 00
      IF(NERROR.NE.0) THEN
        WRITE(6,*) ' Problem in input to ASSEMBLE_MO_FROM_FRAG... '
        NTEST = 100
      END IF
*
      IF(NTEST.GT.0) THEN
        WRITE(6,*) ' Info on mapping from fragment to GAS MOs:'
        WRITE(6,*) ' ========================================= '
        WRITE(6,*)
        DO IFRAG = 1, NFRAG_MOL
          IF(IREAD_NEW_OR_OLD.EQ.1) THEN
           NSMOB_L = NSMOB_FRAG(IFRAG)
          ELSE
           NSMOB_L = NSMOB
          END IF
          WRITE(6,'(A,I3)') 
     &    ' Number of orbitals per GAS(row) and SYM(col) for fragment',
     &      IFRAG
          CALL IWRTMA
     &    (N_GS_SM_BAS_FRAG(0,1,IFRAG),NGAS+2,NSMOB,MXPNGAS+1,MXPOBS)
        END DO
      END IF
*
      IF(NERROR.NE.0) THEN   
        WRITE(6,*) ' Input error in ASSEMBLE_MO_FROM_FRAG... '
        STOP ' Input error in ASSEMBLE_MO_FROM_FRAG... '
      END IF
*
      RETURN
      END
      SUBROUTINE ASSEMBLE_MO_FROM_FRAGMENTS
*
* A set of fragment MO's have been defined to LUCIA
* obtain set of MO's by combining the MO's of the fragments
*
* It is assumed and required that the fragments are given in the 
* same order as in the DALTON calculation on the whole molecule
*
      RETURN
      END
      SUBROUTINE ALLO_ALLO_0
*
* Allocation of a few arrays needed for calculation dimenensions of arrays...
*
*. Jeppe Olsen, Jan. 2012 in Odense
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'spinfo.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'glbbas.inc'
*
*  =================================
*. Information on occupation classes
*  =================================
*
*. Occupations
      CALL MEMMAN(KIOCCLS,NGAS*NOCCLS_MAX,'ADDS  ',1,'IOCCLS')
      DO ISM = 1, NIRREP
        CALL MEMMAN(KIB_OCCLS(ISM),NOCCLS_MAX,'ADDS  ',1,'IB_OCC')
      END DO
*. And basespace for the various occupation classes
      CALL MEMMAN(KBASSPC,NOCCLS_MAX,'ADDS  ',1,'BASSPC')
*. Number of Confs, CSF's, CM's, SD's per occupation class for given sym
      CALL MEMMAN(KNCN_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NCNOCC')
      CALL MEMMAN(KNCS_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NCSOCC')
      CALL MEMMAN(KNCM_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NCMOCC')
      CALL MEMMAN(KNSD_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NSDOCC')
*. Number of String (AB) combinations and determinants -
*. Differ from the number of determinants generated from 
*. configurations, if MS < S
      CALL MEMMAN
     &(KNCMAB_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NCMAOC')
      CALL MEMMAN
     &(KNSDAB_FOR_OCCLS,NIRREP*NOCCLS_MAX,'ADDS  ',1,'NSDAOC')
*
      LEN = NIRREP*(MAXOP+1)*NOCCLS_MAX
C?    WRITE(6,*) ' MAXOP, NOCCLS_MAX, LEN = ',
C?   &             MAXOP, NOCCLS_MAX, LEN
      CALL MEMMAN(KNCN_PER_OP_SM,LEN,'ADDS  ',1,'NCNPOS')
*. Number of confs, all sym for an occupation class
      CALL MEMMAN(KNCN_ALLSYM_FOR_OCCLS,NOCCLS_MAX,'ADDS  ',1,'CN_OCC')
*. Array: AB supergroups => occupation class
      CALL MEMMAN(KIABSPGP_FOR_OCCLS,N_ABSPGP_MAX*2,'ADDL  ',2,
     &            'ABTOOC')
      CALL MEMMAN(KNABSPGP_FOR_OCCLS,NOCCLS_MAX,'ADDL  ',2,'NABTOC')
      CALL MEMMAN(KIBABSPGP_FOR_OCCLS,NOCCLS_MAX,'ADDL  ',2,'BABTOC')
*. AB supergroups for compound space
      CALL MEMMAN(KIABSPGP_FOR_CMPSPC,N_ABSPGP_MAX*2,'ADDL  ',2,
     &            'ABTOCM')
*. Active occupation classes in a given CI space for Sigma and C
      CALL MEMMAN(KCIOCCLS_ACT,NOCCLS_MAX,'ADDL  ',2,'S_OCAC')
      CALL MEMMAN(KSIOCCLS_ACT,NOCCLS_MAX,'ADDL  ',2,'C_OCAC')
*
* Occupation subclasses
*
*. Gas space and occupation of each suboccupation class
      CALL MEMMAN(KOGOCSBCLS,2*NOCSBCLST,'ADDL  ',2,'OGOCSB')
*. The occupation sub classes of each occupation class
      CALL MEMMAN(KOCSBCLS_OF_OCCLS,NOCCLS_MAX*MXPNGAS,'ADDL  ',2,
     &            'OC-SBO')
*. The minumum number of open shells for each occupation subclass in a 
*. given occupation class
      CALL MEMMAN(KMINOPGAS_FOR_OCCLS,NOCCLS_MAX*MXPNGAS,'ADDL  ',2,
     &            'MNGSOC')
*. And the minimum number of open orbitals for each occupation sub class
      CALL MEMMAN(KMNOPOCSBCL,NOCSBCLST,'ADDL  ',2,'MNOPSB')
*. 
*. Subconfigurations
*
C?    WRITE(6,*) ' ALLO_ALLO_0: NSMOB*(MAXOP+1)*NOCSBCLST = ',
C?   &                          NSMOB*(MAXOP+1)*NOCSBCLST
*. Number of subconfigurations per sym, nopen, and subconftype
*
      LL = NSMOB*(MAXOP+1)*NOCSBCLST 
      CALL MEMMAN(KNSBCNF,LL,'ADDL  ',2,' NSBCN')
      CALL MEMMAN(KLSBCNF,NOCSBCLST,'ADDL  ',2,' LSBCN')
      CALL MEMMAN(KIBSBCNF,LL,'ADDL  ',2,'IBSBCN')
*. Pointer to pointers for occupation of subconfigurations of given  occupation subclass
      CALL MEMMAN(KKOCSBCNF,NOCSBCLST,'ADDL  ',2,'KKOCSB')
*
      RETURN
      END
      SUBROUTINE FILEMAN_MINI(IFILE,ITASK)
*
* Initial routine for handling some scratch files
* The routines in LUCIA by Andreas are too fancy for me....
*
* Jeppe Olsen, Feb. 2012, Geneva
*  
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'clunit.inc'
*. Input
      CHARACTER*6 ITASK
* ITASK = ASSIGN => Find a free superscratchfile, reserve, set IFILE to 
*                   this value
* ITASK = FREE   => Free superscratchfile IFILE
* ITASK = PRINT  => Print reserved and available superscratch files
*
      IPRINT = 0
      IF(ITASK(1:6).EQ.'ASSIGN') THEN
        IFILE = 0
        DO ISCR = 1, MXPNSCRFIL
          IF(ISTAT_SUPSCR(ISCR).EQ.0) THEN
* Available file 
           IFILE = LU_SUPSCR(ISCR)
           ISTAT_SUPSCR(ISCR) = 1
           GOTO 1010
          END IF
        END DO
 1010   CONTINUE
        IF(IFILE.EQ.0) THEN
          WRITE(6,*) ' No available superscratchfiles '
          STOP       ' No available superscratchfiles '
        END IF
      ELSE IF (ITASK(1:4).EQ.'FREE') THEN
        DO ISCR = 1, MXPNSCRFIL
         IF(LU_SUPSCR(ISCR).EQ.IFILE) ISTAT_SUPSCR(ISCR)  = 0
        END DO
      END IF
*
      IF(ITASK(1:5).EQ.'PRINT '.OR.IPRINT.EQ.1) THEN
        WRITE(6,*) ' Superscratcfiles, number and use '
        WRITE(6,*) ' ================================='
        DO ISCR = 1, MXPNSCRFIL
         WRITE(6,'(3X, 3(8X,I3))') 
     &   ISCR, LU_SUPSCR(ISCR), ISTAT_SUPSCR(ISCR)
        END DO
      END IF
*
      RETURN
      END
      SUBROUTINE CHECK_ICBLTP
*
* Routine for locating ICLBTP problem
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc-static.inc'
      INCLUDE 'glbbas.inc'
      WRITE(6,*)  ' KCBLTP = ', KCBLTP
      WRITE(6,*)  ' WORK(KCBLTP) as integers:'
      CALL IWRTMA(WORK(KCBLTP),1,4,1,4)
*
      RETURN
      END
C     GET_NCONF_PER_OPEN_FOR_SUM_OCCLS(NCONF_PER_OPEN(1,ISM),
C    &     MAXOP,NOCCLS_ACT,IOCCLS_ACT,ISM,WORK(KNCN_PER_OP_SM),
C    &     NIRREP)

      SUBROUTINE GET_NCONF_PER_OPEN_FOR_SUM_OCCLS(NCONF_PER_OPEN_ACT,
     &           MAXOP,NOCCLS_ACT,IOCCLS_ACT,ISM,
     &           NCN_PER_OP_SM,NIRREP)
*
* Obtain number of configurations per number of open orbitals for 
* a sum of occupation classes.
* 
* Jeppe Olsen, March 2012, for getting CSFs working with multiple 
*              CI spaces
*
      INCLUDE 'implicit.inc'
*. Input
      INTEGER NCN_PER_OP_SM(MAXOP+1,NIRREP,*),
     &        IOCCLS_ACT(NOCCLS_ACT)
*. Output
      INTEGER NCONF_PER_OPEN_ACT(MAXOP+1)
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Output from GET_NCONF_PER_OP... '
      END IF
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' ISM, NOCCLS_ACT = ', ISM, NOCCLS_ACT
        WRITE(6,*) ' NIRREP, MAXOP = ', NIRREP, MAXOP
      END IF
*
      IZERO = 0
      CALL ISETVC(NCONF_PER_OPEN_ACT,IZERO,MAXOP+1)
      DO IIOCCLS = 1, NOCCLS_ACT
        IOCCLS = IOCCLS_ACT(IIOCCLS)
        IF(NTEST.GE.1000) 
     &  WRITE(6,*) ' IOCCLS, IIOCCLS = ', IOCCLS, IIOCCLS
        IONE = 1
        CALL IVCSUM(NCONF_PER_OPEN_ACT,NCONF_PER_OPEN_ACT,
     &              NCN_PER_OP_SM(1,ISM,IOCCLS),IONE,IONE,
     &              MAXOP + 1)
      END DO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' NCONF_PER_OPEN_ACT array (output) '
        CALL IWRTMA(NCONF_PER_OPEN_ACT,1,MAXOP+1,1,MAXOP+1)
      END IF
*
      RETURN
      END
      SUBROUTINE SIGDEN_CIGEN(CB,HCB,LUC,LUHC,ITASK)
*
* Outer routine for sigma, traci, densi  generation
* GAS version 
*
* IF ICISTR.gt.1, then CB, HCB are two blocks holding a batch
* IF ICISTR .eq. 1, then CB, HCB are two vectors holding a vector over
* parameters. Parameters are CSF's if required
*
* IF CSF's are active (NOCSF = 0), then three vectors over SD's 
* must be available (KCOMVECX_SD, X = 1, 2, 3)
*
*
* A new start, March 23, 2012
*
* Last modification; Oct. 30, 2012; Jeppe Olsen; changed call to Z_BLKFO
      INCLUDE 'wrkspc.inc'
      CHARACTER*6 ITASK
*
* =====
*.Input
* =====
*
*.Definition of c and sigma
      INCLUDE 'cands.inc'
*
*./ORBINP/: NACOB used
      INCLUDE 'orbinp.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'strbas.inc'
      INCLUDE 'cstate.inc' 
      INCLUDE 'strinp.inc'
      INCLUDE 'stinf.inc'
      INCLUDE 'csm.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'gasstr.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cprnt.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'oper.inc'
      INCLUDE 'cecore.inc'
      INCLUDE 'spinfo.inc'
      COMMON/CMXCJ/MXCJ,MAXK1_MX,LSCMAX_MX
*. Two blocks of C or Sigma (for ICISTR .gt. 2)
      DIMENSION CB(*),HCB(*)
*. Two vectors of C or Sigma (for ICISTR = 1)
*
      CALL QENTER('SGDEGN')
*
      
      NTEST = 010
      IF(NTEST.GE.10) THEN
        WRITE(6,*) 
        WRITE(6,*) 'SIGDEN_CIGEN speaking '
        WRITE(6,*) '===================== '
        WRITE(6,*) 
        WRITE(6,'(A,A6)') ' ITASK = ', ITASK
      END IF 
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Input to SIGDEN_CIGEN '
        WRITE(6,*) ' ======================='
        IF(ICISTR.GT.1) THEN
          CALL WRTVCD(CB,LUC,1,-1)
        ELSE
          CALL WRTMAT(CB,1,NCVAR,1,NCVAR)
        END IF
      END IF
C?    WRITE(6,*) ' Ecore = ', ECORE
        
      IDUM = 0
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'SGDEGN')
*. For the moment
      ICFIRST = 1
      ISFIRST = 1
*
      MAXK1_MX = 0
      LSCMAX_MX = 0
      IF(ISSPC.LE.NCMBSPC) THEN
        IATP = 1
        IBTP = 2
      ELSE
        IATP = IALTP_FOR_GAS(ISSPC)
        IBTP = IBETP_FOR_GAS(ISSPC)
      END IF
*
      NOCTPA = NOCTYP(IATP)
      NOCTPB = NOCTYP(IBTP)
*. Offset for supergroups 
      IOCTPA = IBSPGPFTP(IATP)
      IOCTPB = IBSPGPFTP(IBTP)
*
      NAEL = NELEC(IATP)
      NBEL = NELEC(IBTP)
*. Block for storing complete or partial CI-vector
      IF(ISIMSYM.EQ.0) THEN
        LBLOCK = MXSOOB
      ELSE
        LBLOCK = MXSOOB_AS
      END IF
      IF(NOCSF.EQ.0) THEN
        LBLOCK  = NSD_FOR_OCCLS_MAX
      END IF
      LBLOCK = MAX(LBLOCK,LCSBLK)
C?    WRITE(6,*) ' TEST, MV7: LCSBLK, LBLOCK = ', LCSBLK, LBLOCK
      ICOMP = 0
      ILTEST = -3006
      IF(ISFIRST.EQ.1) THEN
        CALL Z_BLKFO_FOR_CISPACE(ISSPC,ISSM,LBLOCK,ICOMP,
     &       NTEST,NSBLOCK,NSBATCH,
     &       dbl_mb(KSIOIO),int_mb(KSBLTP),NSOCCLS_ACT,dbl_mb(KSIOCCLS_ACT),
     &       int_mb(KSLBT),int_mb(KSLEBT),int_mb(KSLBLK),int_mb(KSI1BT),
     &       int_mb(KSIBT),
     &       int_mb(KSNOCCLS_BAT),int_mb(KSIBOCCLS_BAT),ILTEST)
      END IF
      IF(ICFIRST.EQ.1) THEN
        CALL Z_BLKFO_FOR_CISPACE(ICSPC,ICSM,LBLOCK,ICOMP,
     &       NTEST,NCBLOCK,NCBATCH,
     &       int_mb(KCIOIO),int_mb(KCBLTP),NCOCCLS_ACT,dbl_mb(KCIOCCLS_ACT),
     &       int_mb(KCLBT),int_mb(KCLEBT),int_mb(KCLBLK),int_mb(KCI1BT),
     &       int_mb(KCIBT),
     &       int_mb(KCNOCCLS_BAT),int_mb(KCIBOCCLS_BAT),ILTEST)
      END IF
C     WRITE(6,*) ' ECORE in MV7 =', ECORE
*. Number of BLOCKS
        NBLOCK = NSBLOCK
C?      WRITE(6,*) ' Number of blocks ', NBLOCK

      IF(I12.EQ.2) THEN
        IDOH2 = 1
      ELSE
        IDOH2 = 0
      END IF
*
      IF(NOCSF.EQ.0.AND.ICNFBAT.GE.2) THEN
*. Obtain scratch files for saving combination forms of C and Sigma
C             FILEMAN_MINI(IFILE,ITASK)
         CALL FILEMAN_MINI(LU_CDET,'ASSIGN')
         CALL FILEMAN_MINI(LU_SDET,'ASSIGN')
C?       WRITE(6,*) ' Test: LU_CDET, LU_SDET: ',
C?   &                      LU_CDET, LU_SDET
* ITASK = ASSIGN => Find a free superscratchfile, reserve, set IFILE to 
* ITASK = FREE   => Free superscratchfile IFILE
      END IF
*
      IF(ICISTR.EQ.1) THEN
       LLUC = 0
       LLUHC = 0
      ELSE 
       IF(NOCSF.EQ.1) THEN
        LLUC = LUC
        LLUHC = LUHC
       ELSE
        LLUC = LU_CDET
        LLUHC = LU_SDET
       END IF
      END IF

      IF(NOCSF.EQ.0) THEN
       IF(ICNFBAT.EQ.1) THEN
*. In core
         CALL CSDTVCM(CB,WORK(KCOMVEC1_SD),WORK(KCOMVEC2_SD),
     &                1,0,ICSM,ICSPC,2)
       ELSE
*. Not in core- write determinant expansion on LU_CDET 
C       CSDTVCMN(CSFVEC,DETVEC,SCR,IWAY,ICOPY,ISYM,ISPC,
C    &           IMAXMIN_OR_GAS,ICNFBAT,LU_DET,LU_CSF,NOCCLS_ACT,
C    &           IOCCLS_ACT,IBLOCK,NBLK_PER_BATCH)  
        CALL CSDTVCMN(CB,HCB,WORK(KVEC3),
     &       1,0,ICSM,ICSPC,2,2,LU_CDET,LUC,NCOCCLS_ACT,
     &       WORK(KCIOCCLS_ACT),int_mb(KCIBT),int_mb(KCLBT))
       END IF
      END IF
*
C            RASSG3(CB,SB,LBATS,LEBATS,I1BATS,IBATS,LUC,LUHC,C,HC,ECORE)
      IF(ICISTR.GE.2) THEN
        CALL RASSG3(CB,HCB,NSBATCH,int_mb(KSLBT),int_mb(KSLEBT),
     &       int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,XDUM,XDUM,ECORE,
     &       ITASK)
      ELSE
*. ICISTR = 1, CB, HCB are the complete vectors
        IF(NOCSF.EQ.1) THEN
*. CB and HCB on input are the complete vectors
          CALL RASSG3(WORK(KVEC1P),WORK(KVEC2P),NSBATCH,
     &         int_mb(KSLBT),int_mb(KSLEBT),
     &         int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,CB,HCB,ECORE,
     &         ITASK)
*. Input is in KCOMVEC1_SD, construct output in KCOMVEC2_SD
        ELSE
          CALL RASSG3(WORK(KVEC1P),WORK(KVEC2P),NSBATCH,
     &         int_mb(KSLBT),int_mb(KSLEBT),
     &         int_mb(KSI1BT),int_mb(KSIBT),LLUC,LLUHC,
     &         WORK(KCOMVEC1_SD),WORK(KCOMVEC2_SD),ECORE,
     &         ITASK)
        END IF ! CSF switch
      END IF ! ICISTR switch
*
      IF(NOCSF.EQ.0) THEN
* Transform sigma vector in KCOMVEC2_SD to CSF basis
       IF(ICNFBAT.EQ.1) THEN
C CSDTVCM(CSFVEC,DETVEC,IWAY,ICOPY,ISYM,ISPC,IMAXMIN_OR_GAS)
         CALL CSDTVCM(HCB,WORK(KCOMVEC2_SD),WORK(KCOMVEC1_SD),
     &        2,0,ISSM,ISSPC,2)
       ELSE
        CALL CSDTVCMN(HCB,CB,WORK(KVEC3),
     &       2,0,ISSM,ISSPC,2,2,LU_SDET,LUHC,NSOCCLS_ACT,
     &       WORK(KSIOCCLS_ACT),WORK(KSIBT),int_mb(KSLBT))
       END IF
      END IF
*
      IF(NOCSF.EQ.0.AND.ICNFBAT.GE.2) THEN
        CALL FILEMAN_MINI(LU_CDET,'FREE  ')
        CALL FILEMAN_MINI(LU_SDET,'FREE  ')
      END IF
*
      IF(NTEST.GE.1000) THEN
        IF(ITASK(1:5).EQ.'SIGMA'. OR. ITASK(1:5).EQ.'TRACI') THEN
         WRITE(6,*) ' Output vector from SGDEGN '
         WRITE(6,*) ' ========================= '
         IF(ICISTR.GT.1) THEN
           CALL WRTVCD(CB,LUHC,1,-1)
         ELSE 
           CALL WRTMAT(HCB,1,NSVAR,1,NSVAR)
         END IF
        END IF
      END IF
*
*. Eliminate local memory
      CALL MEMMAN(KDUM ,IDUM,'FLUSM ',2,'SGDEGN')
*
      CALL QEXIT('SGDEGN')
*
      RETURN
      END
      SUBROUTINE BLK_SET_REORDER_XMAT(X,NBLK,LBLK,IREO)
*
* Reorderings, IREO(I) of index I, are given
* Set up the corresponding real permutation matrix
* giving the reordered vectors in terms of the original,
* i.e. X(I,J) = 1.0D0 if J = IREO(I)
*
*. Jeppe Olsen, May 27, 2012
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
*. Input 
      INTEGER LBLK(NBLK), IREO(*)
*. Output
      DIMENSION X(*)
*
      NTEST = 0
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' Info from BLK_SET_REORDER_XMAT '
        WRITE(6,*) ' ============================= '
      END IF
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Dimension of each block'
        CALL IWRTMA3(LBLK,1,NBLK,1,LBLK)
      END IF
*. Total dimension
      NDIM = IELSUM(LBLK,NBLK)
      IF(NTEST.GE.10) THEN
        WRITE(6,*) ' Reorder array '
        CALL IWRTMA3(IREO,1,NDIM,1,NDIM)
      END IF
*
      IB_I = 1
      IB_IJ = 1
      DO IBLK = 1, NBLK
       L = LBLK(IBLK)
*. Reorder block IBLK
       CALL SET_REORDER_XMAT(X(IB_IJ),L,IREO(IB_I),IB_I)
       IB_IJ = IB_IJ + L**2
       IB_I  = IB_I  + L
      END DO! Loop over blocks
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Reordering matrix '
        CALL APRBLM2(X,LBLK,LBLK,NBLK,0)
      END IF
*
      RETURN
      END
      SUBROUTINE SET_REORDER_XMAT(X,NDIM,IREO,IOFF)
*
* Reorderings, IREO(I)-IOFF + 1 of index I, are given
* Set up the corresponding real permutation matrix
* giving the reordered vectors in terms of the original,
* i.e. X(I,J) = 1.0D0 if J = IREO(I) - IOFF + 1
*
      INCLUDE 'implicit.inc'
*. Input
      DIMENSION IREO(NDIM)
*. Output
      DIMENSION X(*)
*
      ZERO = 0.0D0
      CALL SETVEC(X,ZERO,NDIM**2)
*
      DO I = 1, NDIM
        J = IREO(I) - IOFF + 1
C       WRITE(6,*) ' Testy, I, J = ', I,J
        X((J-1)*NDIM + I) = 1.0D0
      END DO
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Generated reorder matrix '
        CALL WRTMAT(X,NDIM,NDIM,NDIM,NDIM)
      END IF
*
      RETURN
      END
      SUBROUTINE MOINF_GN_LUCIA(IFRAG,LUCINF)
*
* READ in information about fraction IFRAG from  LUCIA
* interface file LUCINF. Save in in arrays for fragment IFRAG
* CMOAO_FRAG matrix is read to KKCMOAO_FRAG(IFRAG) which is also allocated 
* here
* 
* LUCINF is pt a 8 character string with name LUCINF_0, X = 0, 9 
* 
* corresponds to complete molecule
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc-static.inc'
      INCLUDE 'irat.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'fragmol.inc'
      INCLUDE 'lucinp.inc'
      CHARACTER*8 LUCINF
*. Scratch
      INTEGER ISCR(MXPOBS)
*
      NTEST = 00
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MOINF_GN_LUCIA for IFRAG = ', IFRAG
        WRITE(6,*) ' =================================='
        WRITE(6,*)
        WRITE(6,'(A,A)') ' LUCINF: ', LUCINF
      END IF
*
      LUNUMB = 16
      OPEN(LUNUMB,STATUS='OLD',FORM='FORMATTED',FILE=LUCINF)
*
      REWIND LUNUMB
*
      ZERO = 0.0D0
      CALL SETVEC(NBAS_FRAG(1,IFRAG),IZERO,MXPOBS)
      READ(LUNUMB,*) NSYMF
      NSYM_FRAG(IFRAG) = NSYMF
*. Number of basis functions per sym
      READ(LUNUMB,*) (NBAS_FRAG(I,IFRAG),I=1,NSYMF)
*. Number of MO's per sym
      READ(LUNUMB,*) (ISCR(I),I=1,NSYMF)
*. Number of Mo coefficients
      READ(LUNUMB,*) NCMOF
* NCMO may also be calculated from NBAS_FRAG. Check
      NDIM_CFRAG = NDIM_1EL_MAT(1,NBAS_FRAG(1,IFRAG),NBAS_FRAG(1,IFRAG),
     &             NSYMF,0)
      IF(NDIM_CFRAG.NE.NCMOF) THEN 
        WRITE(6,*) 
     &  ' Difference between computed and read dimension of CMO_FRAG '
        WRITE(6,*) ' NDIM_CFRAG, NCMOF = ', NDIM_CFRAG,NCMOF
        WRITE(6,*) ' NSYMF = ', NSYMF
        STOP       
     &  ' Difference between computed and read dimension of CMO_FRAG '
      END IF
      IF(NTEST.GE.100) WRITE(6,*) ' Number MO-AO coefs' , NDIM_CFRAG
*. Allocate memory and read MO-AO transformation matrix
      CALL MEMMAN(KCMOAO_FRAG(IFRAG),NDIM_CFRAG,'ADDL  ',2,'MOAO_F')
      READ(LUNUMB,*) (WORK(KCMOAO_FRAG(IFRAG)-1+I),I=1, NDIM_CFRAG)
*. Read in the  AO centers and types
      NBAST = IELSUM(NBAS_FRAG(1,IFRAG),NSYMF)
      IF(NTEST.GE.100)
     &WRITE(6,*) ' TESTY, NSYMF, NBAST = ', NSYMF, NBAST
      READ(LUNUMB,'(20A4)') (AO_CENT_FRAG(IAO,IFRAG),IAO = 1, NBAST)
      READ(LUNUMB,'(20A4)') (AO_TYPE_FRAG(IAO,IFRAG),IAO = 1, NBAST)
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' AO_CENT and AO_TYPE for fragment ', IFRAG
        WRITE(6,'(20A4)') (AO_CENT_FRAG(IAO,IFRAG),IAO = 1, NBAST)
        WRITE(6,'(20A4)') (AO_TYPE_FRAG(IAO,IFRAG),IAO = 1, NBAST)
      END IF
      CLOSE(LUNUMB,STATUS='KEEP')
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MO- AO transformation matrix for fragment', IFRAG  
        WRITE(6,*) ' ================================================'
        CALL APRBLM_F7(WORK(KCMOAO_FRAG(IFRAG)),
     &       NBAS_FRAG(1,IFRAG),NBAS_FRAG(1,IFRAG),NSYMF,0)
        CALL PRINT_CMOAO_FRAG(WORK(KCMOAO_FRAG(IFRAG)),IFRAG)
      END IF
*
      RETURN
      END
      SUBROUTINE NATORB3_GS(RHO1,XNAT,RHO1_GS,OCCNUM,
     &                  SCR,IREO_GS_TO_TS,IPRDEN)
*
* Obtain natural orbitals for general symmetry blocks. Input density is 
* over orbitals on GASpaces (active orbitals) in type-symmetry order
*
* Jeppe Olsen, July 2012, from NATORB3
*              
* Last modification, July 8, 2012 (Jeppe) 
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cgas.inc'
*. Input
      DIMENSION RHO1(NACOB,NACOB), IREO_GS_TO_TS(NACOB)
*. Output
      DIMENSION RHO1_GS(*),OCCNUM(*),XNAT(*)
*. Scratch ( Largest symmetry block )
      DIMENSION SCR(*)
*. Local scratch
      INTEGER NACOB_GS(MXP_NSUPSYM)
*
      NTESTL = 0
      NTEST = MAX(NTESTL,IPRDEN)
      IF(NTEST.GE.100) WRITE(6,*) ' Info from NATORB3_GS'
*
* Number of active orbitals per general symmetry
*
C?    WRITE(6,*) ' NGENSMOB = ', NGENSMOB
      DO IGENSM = 1, NGENSMOB
       NACOB_GS(IGENSM) = 0
       DO IGAS = 1, NGAS
         NACOB_GS(IGENSM) = NACOB_GS(IGENSM) + NGAS_GNSYM(IGENSM,IGAS)
C?       WRITE(6,*) ' IGENSM, IGAS, NGAS_GNSYM = ',
C?   &                IGENSM, IGAS, NGAS_GNSYM(IGENSM,IGAS)
       END DO
      END DO
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Number of active orbitals per general sym '
        CALL IWRTMA3(NACOB_GS,1,NGENSMOB,1,NGENSMOB)
      END IF
*
* Reform density to blocks over general symmetry
*
C           REFORM_RHO1_TO_GNSM(RHO1_ST,RHO1_GNSM_ST,IWAY,IREO_GNSYM_TO_TS)
C?    WRITE(6,*) ' IREO_GS_TO_TS: '
C?    CALL IWRTMA3(IREO_GS_TO_TS,1,NACOB,1,NACOB)
      CALL REFORM_RHO1_TO_GNSM(RHO1,RHO1_GS,1,IREO_GS_TO_TS)
      IF(NTEST.GE.2 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' Density matrix over general symmetry blocks  '
        WRITE(6,*) ' ============================================='
        WRITE(6,*)
        CALL APRBLM2(RHO1_GS,NACOB_GS,NACOB_GS,NGENSMOB,0)
      END IF
*. Scale with -1 to get natural orbitals with highest occs out first
      LEN_GS = LEN_BLMAT(NGENSMOB,NACOB_GS,NACOB_GS,0)
C                         NBLK,LROW,LCOL,IPACK)
      ONEM = -1.0D0
      CALL SCALVE(RHO1_GS,ONEM,LEN_GS)
*. And diagonalize
C   DIAG_BLK_SYMMAT(A,NBLK,LBLK,X,EIGENV,SCR,ISYM)
      CALL DIAG_BLK_SYMMAT(RHO1_GS,NGENSMOB,NACOB_GS,XNAT,OCCNUM,SCR,0)
*. And scale occupation numbers back to sense
      CALL SCALVE(OCCNUM,ONEM,NACOB)
*. It could be that degenerate occupations should be scaled
*. to max diagonal elements
*. interchange orbital IOB and IOB -1
      IF(NTEST.GE.1) THEN
        WRITE(6,*)
        WRITE(6,*) 
     &  ' Natural occupation numbers for general symmetry '
        WRITE(6,*)
     &  ' ================================================='
        WRITE(6,*)
        CALL PRINT_SCALAR_PER_ORB2(OCCNUM,NACOB_GS,NGENSMOB)
      END IF
      IF(NTEST.GE.5 ) THEN
        WRITE(6,*)
        WRITE(6,*) ' Eigenvectors of natural orbitals(MO-MO) '
        WRITE(6,*)
        CALL APRBLM2(XNAT,NACOB_GS,NACOB_GS,NGENSMOB,0)
      END IF
*
      RETURN
      END 
      SUBROUTINE MOROT_GS(IMO)
*
* A MO-MO rotation matrix is given in KMOMO. Obtain
* final MO-MO rotation matrix by defining internal rotations
*
* General symmetry version, July 2012, Jeppe Olsen
*
*                Sept. 26 2012, Jeppe Olsen-Debugged
* Last revision, March 9, 2013, Jeppe Olsen, subshell averaging of densities added
*
* Type of active orbitals is provided by the keyword IMO
*
* IMO = 1 => Natural orbitals
* IMO = 2 => Canonical orbitals
* IMO = 3 => Pseudo-natural orbitals
* IMO = 4 => Pseudo-canonical orbitals
* IMO = 5 => Psedo-natural-canonical orbitals
*
* The inactive and secondary orbitals are in general defined
* as pseudo-canonical orbitals
*
* Expansion of current MO's in initial MO's is assumed in KMOMO
* Final MO-AO expansion stored in KMOAO
*       MO-MO expansion stored in KMOMO
*
* If no mo-ao trans is present, only MOMO matrix is returned
*
*. Note: In case of super-symmetry one has to destinguish between two ways 
*. of having the orbitals arranged for a given standard symmetry: the actual/gas
*. order or standard order. The routine returns the MO coefficients in actual/gas order
      INCLUDE 'wrkspc.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'glbbas.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'cgas.inc'
      INCLUDE 'cintfo.inc'
*. Local scratch
      INTEGER NACOB_GS(MXP_NSUPSYM),NTOOB_GS(MXP_NSUPSYM)
*
      NTEST = 10
      IF(NTEST.GE.10) THEN
        WRITE(6,*)
        WRITE(6,*) ' =================='
        WRITE(6,*) ' MOROT_GS in action '
        WRITE(6,*) ' =================='
        WRITE(6,*)
      END IF
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' IMO parameter ', IMO
        WRITE(6,*) ' INTIMP = ', INTIMP
      END IF
      IF(NTEST.GE.1) THEN
        IF(IMO.EQ.1) THEN
          WRITE(6,*) ' Final orbitals: natural orbitals '
        ELSE IF (IMO.EQ.2) THEN
          WRITE(6,*) ' Final orbitals: canonical orbitals '
        ELSE IF (IMO.EQ.3) THEN
          WRITE(6,*) ' Final orbitals: pseudo-natural orbitals '
        ELSE IF (IMO.EQ.4) THEN
          WRITE(6,*) ' Final orbitals: pseudo-canonical orbitals '
        ELSE IF (IMO.EQ.5) THEN
          WRITE(6,*) 
     &    ' Final orbitals: pseudo-natural-canonical orbitals'
        END IF
      END IF
*
*. Number of orbitals and active orbitals per standard symmetry
*
      DO IGENSM = 1, NGENSMOB
        N = 0
        DO IGAS = 1, NGAS
          N = N + NGAS_GNSYM(IGENSM,IGAS)
        END DO
        NACOB_GS(IGENSM) = N
        NTOOB_GS(IGENSM) = 
     &  N + NGAS_GNSYM(IGENSM,0) + NGAS_GNSYM(IGENSM,NGAS+1)
      END DO
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' NGENSMOB== ', NGENSMOB
        WRITE(6,*) ' NTOOB_GS=='
        CALL IWRTMA3(NTOOB_GS,1,NGENSMOB,1,NGENSMOB)
        WRITE(6,*) ' NACOB_GS=='
        CALL IWRTMA3(NACOB_GS,1,NGENSMOB,1,NGENSMOB)
      END IF
*
      IF(NTEST.GE.1000) THEN
       WRITE(6,*) 
     & ' Initial MOMO matrix over blocks of standard symmetry'
       CALL APRBLM2(WORK(KMOMO),NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      CALL MEMMAN(IDUM,IDUM,'MARK  ',IDUM,'MOROT ')
      CALL MEMMAN(KLMAT1,NTOOB**2,'ADDL  ',2,'MAT1  ')
      CALL MEMMAN(KLMAT2,NTOOB**2,'ADDL  ',2,'MAT2  ')
      CALL MEMMAN(KLMAT2C,NTOOB**2,'ADDL  ',2,'MAT2C ')
      CALL MEMMAN(KLMAT3,NTOOB**2,'ADDL  ',2,'MAT3  ')
      CALL MEMMAN(KLMAT4,2*NTOOB**2,'ADDL  ',2,'MAT4  ')
      CALL MEMMAN(KLMAT5,NTOOB**2,'ADDL  ',2,'MAT5  ')
      CALL MEMMAN(KLFIFA_GS,NTOOB**2,'ADDL  ',2,'FIFAGS')
      CALL MEMMAN(KLLEN_GASGS_BLK,(NGAS+2)*NGENSMOB,'ADDL  ',1,'LGASGS')
      CALL MEMMAN(KLLEN_GASGS_AC_BLK,(NGAS+2)*NGENSMOB,'ADDL  ',
     &            1,'LGASGS')
      CALL MEMMAN(KLMOMO_GN,NTOOB**2,'ADDL  ',2,'MOMOGN')
*
      LMOMO = NDIM_1EL_MAT(1,NTOOBS,NTOOBS,NSMOB,0)
      LACAC = NDIM_1EL_MAT(1,NACOBS,NACOBS,NSMOB,0)
      LACAC_GS =  LEN_BLMAT(NGENSMOB,NACOB_GS,NACOB_GS,1)
      LMOMO_GS = LEN_BLMAT(NGENSMOB,NTOOB_GS,NTOOB_GS,0)
C LEN_BLMAT(NBLK,LROW,LCOL,IPACK)
*. Number and length of gas- general symmetry blocks
C  LEN_GAS_GS_BLOCKS(LEN_GAS_GS,N_GAS_GS)
      CALL LEN_GAS_GS_BLOCKS(WORK(KLLEN_GASGS_BLK),NGAS_GS_BLKS,
     &                       0,NGAS+1)
      CALL LEN_GAS_GS_BLOCKS(WORK(KLLEN_GASGS_AC_BLK),NGAS_GS_AC_BLKS,
     &                       1,NGAS)
*
C?    WRITE(6,*) ' LMOMO, LACAC = ', LMOMO, LACAC
C     NDIM_1EL_MAT(IHSM,NRPSM,NCPSM,NSM,IPACK)
*
* We will in the following do reordering from actual to supersymmetry-
* blocked forms. Put the corresponding array in place
      CALL ISWPVE(IACT_TO_GENSM_REO, ISTA_TO_GENSM_REO, NTOOB)
*
* Obtain MOMO over general symmetry
C     REFORM_MAT_STA_GEN(ASTA,AGEN,IPACK,IWAY)
      CALL REFORM_MAT_STA_GEN(WORK(KMOMO),WORK(KLMOMO_GN),0,1)
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Initial MOMO over general symmetry '
        CALL APRBLM2(WORK(KLMOMO_GN),NTOOB_GS,NTOOB_GS,NGENSMOB,0)
      END IF
*
* If supersymmetry is active, then average the density matrix over components 
* belonging to the same irrep
*
      IF(I_USE_SUPSYM.EQ.1) THEN
*. Bring first density to form with supersymmetry blocks
C            REFORM_RHO1_TO_GNSM(RHO1_ST,RHO1_GNSM_ST,IWAY,IREO_GNSYM_TO_TS)
        CALL REFORM_RHO1_TO_GNSM(WORK(KRHO1),WORK(KLMAT2),1,
     &       WORK(KIREO_GNSYM_TO_TS_ACOB))
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' Reformed density matrix '
          CALL APRBLM2(WORK(KLMAT2),NACOB_GS,NACOB_GS,NGENSMOB,0)
        END IF
*. Average components density
C       AVE_SUPSYM_MAT(ASUP,NOBPSPSM,IPACK)
        CALL AVE_SUPSYM_MAT(WORK(KLMAT2),NACOB_GS,0)
*. And reform back to standard form of density
        ZERO = 0.0D0
        CALL SETVEC(WORK(KRHO1),ZERO,NACOB**2)
        CALL REFORM_RHO1_TO_GNSM(WORK(KRHO1),WORK(KLMAT2),2,
     &       WORK(KIREO_GNSYM_TO_TS_ACOB))
        IF(NTEST.GE.0000) THEN
         WRITE(6,*) ' Supersymmetry averaged density '
         CALL  WRTMAT(WORK(KRHO1),NACOB,NACOB,NACOB,NACOB)
        END IF
      END IF
*
*. 1: Construct and diagonalize FI + FA matrix 
* =============================================
*
      KINT2 = KINT_2EINI
      IE2ARR_F = IE2LIST_I(IE2LIST_IB(IE2LIST_FULL))
      KINT2_FSAVE = KINT2_A(IE2ARR_F)
      KINT2_A(IE2ARR_F) = KINT_2EINI
      KINT2_A(IE2ARR_F) = KINT_2EMO
*
      CALL FI_FROM_INIINT(WORK(KFI),WORK(KMOMO),WORK(KH),
     &                      ECORE_HEX,1)
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' Inactive Fock matrix '
        CALL APRBLM2(WORK(KFI),NTOOBS,NTOOBS,NSMOB,1)
      END IF
      CALL FA_FROM_INIINT
     &(WORK(KFA),WORK(KMOMO),WORK(KMOMO),WORK(KRHO1),1)
      KINT2_A(IE2ARR_F)  = KINT2_FSAVE
*
      ONE = 1.0D0
      CALL VECSUM(WORK(KLMAT1),WORK(KFI),WORK(KFA),ONE,ONE,NINT1)
      IF(IMO.EQ.5) CALL COPVEC(WORK(KLMAT1),WORK(KLMAT5),NINT1)
*
      IF(NTEST.GE.1000) THEN
        WRITE(6,*) ' FIFA matrix in standard symmetry'
        CALL APRBLM2(WORK(KLMAT1),NTOOBS,NTOOBS,NSMOB,1)
      END IF
* Reform FIFA to general symmetry
C          REFORM_MAT_STA_GEN(ASTA,AGEN,IPACK,IWAY)
      CALL REFORM_MAT_STA_GEN(WORK(KLMAT1),WORK(KLFIFA_GS),1,1)
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' FIFA matrix in general symmetry '
        CALL APRBLM2(WORK(KLFIFA_GS),NTOOB_GS,NTOOB_GS,NGENSMOB,1)
      END IF
*
* Obtain canonical or pseudo canonical orbitals by diagonalization of FIFA
*
* Orbital energues in KLMAT1, eigenvectors in KLMAT2C
      IF(IMO.EQ.2) THEN
*. Diagonalize symmetry blocks
C           DIAG_BLK_SYMMAT(A,NBLK,LBLK,X,EIGENV,SCR,ISYM)
       CALL DIAG_BLK_SYMMAT(WORK(KLFIFA_GS),NGENSMOB,
     &      NTOOB_GS,WORK(KLMAT2C),WORK(KLMAT1),
     &      WORK(KLMAT4),1)
      ELSE
*. Diagonalize symmetry-type blocks 
*. Obtain in KLMAT4 type-symmetry blocks of FIFA
C           EXTR_CP_GASBLKS_FROM_GENSYM_MAT(AS,ASG,IEORC,IGAS_F,IGAS_L,IPAK)
       CALL EXTR_CP_GASBLKS_FROM_GENSYM_MAT(
     &      WORK(KLFIFA_GS),WORK(KLMAT4),2,0,NGAS+1,1)
       CALL DIAG_BLK_SYMMAT(WORK(KLMAT4),NGAS_GS_BLKS,
     &       WORK(KLLEN_GASGS_BLK),WORK(KLMAT2C),WORK(KLMAT1),
     &       WORK(KLMAT4),1)
      END IF !MO = 2 switch
*
* So in output, the expansion of the MO's are in KLMAT2C
*  IMO = 2 => Complete symmetry blocks
*  else    => Symmetry-type blocks
*
      IF(NTEST.GE.56) THEN
         WRITE(6,*) ' Canonical orbital energies'
         CALL PRINT_SCALAR_PER_ORB2(WORK(KLMAT1),NTOOB_GS,NGENSMOB)
      END IF
      IF(NTEST.GE.100) THEN
        WRITE(6,*)  ' Expansion of final orbitals, canonical part'
        IF(IMO.EQ.2) THEN
          IS_OR_SG = 1
        ELSE
          IS_OR_SG = 2
        END IF
C            WRT_SG_MAT(A,IS_OR_SG,IGAS_F,IGAS_L,IPAK,IEXT)
        CALL WRT_SG_MAT(WORK(KLMAT2C),IS_OR_SG,0,NGAS+1,0,1)
      END IF
*
      IF(IMO.EQ.4) THEN
*. Pseudo-canonical orbitals: reform form type-symmetry blocks to 
*. symmetry blocks
*. Expand KLMAT2C to full symmetry blocks
C       EXTR_CP_GASBLKS_FROM_GENSYM_MAT(AS,ASG,IEORC,IGAS_F,IGAS_L,IPAK)
        ZERO = 0.0D0
        CALL SETVEC(WORK(KLMAT4),ZERO,LMOMO_GS)
        CALL EXTR_CP_GASBLKS_FROM_GENSYM_MAT(
     &       WORK(KLMAT4),WORK(KLMAT2C),1,0,NGAS+1,0)
        CALL COPVEC(WORK(KLMAT4),WORK(KLMAT2C),LMOMO_GS)
      END IF
*
* 2: Natural or Pseudo natural orbitals for the active orbitals
* ==============================================================
*
      IF(IMO.EQ.1.OR.IMO.EQ.3.OR.IMO.EQ.5) THEN
*. Obtain and diagonalize symmetry ordered density matrix 
*. over active orbitals
C            REFORM_RHO1_TO_GNSM(RHO1_ST,RHO1_GNSM_ST,IWAY,IREO_GNSYM_TO_TS)
        CALL REFORM_RHO1_TO_GNSM(WORK(KRHO1),WORK(KLMAT2),1,
     &       WORK(KIREO_GNSYM_TO_TS_ACOB))
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' Reformed density matrix '
          CALL APRBLM2(WORK(KLMAT2),NACOB_GS,NACOB_GS,NGENSMOB,0)
        END IF
*. Pack to triangular form
C            TRIPAK_BLKM(AUTPAK,APAK,IWAY,LBLOCK,NBLOCK)
        CALL TRIPAK_BLKM(WORK(KLMAT2),WORK(KLMAT1),1,
     &       NACOB_GS,NGENSMOB)
        ONEM = -1.0D0
        CALL SCALVE(WORK(KLMAT1),ONEM,LACAC_GS)
        IF(NTEST.GE.1000) THEN
          WRITE(6,*) ' Packed density matrix ( times - 1 )'
          CALL APRBLM2(WORK(KLMAT1),NACOB_GS,NACOB_GS,NGENSMOB,1)
        END IF
        IF (IMO.EQ.1.OR.IMO.EQ.5) THEN
*. Diagonalize symmetryblocks of density matrix over active orbitals 
*  and save eigenvectors in KLMAT2
           WRITE(6,*) ' NGENSMOB(1) = ', NGENSMOB
C             DIAG_BLK_SYMMAT(A,NBLK,LBLK,X,EIGENV,SCR,ISYM)
         CALL DIAG_BLK_SYMMAT(WORK(KLMAT1),NGENSMOB,NACOB_GS,
     &        WORK(KLMAT2),WORK(KLMAT3),WORK(KLMAT4),1)
         IF(NTEST.GE.100) THEN
           WRITE(6,*)  ' Expansion of (pseudo-natural) orbitals'
           CALL APRBLM2(WORK(KLMAT2),NACOB_GS,NACOB_GS,NGENSMOB,0)
         END IF
        ELSE IF (IMO.EQ.3) THEN
*. Extract type-symmetry blocks of RHO1
C             EXTR_CP_GASBLKS_FROM_GENSYM_MAT(AS,ASG,IEORC,IGAS_F,IGAS_L,IPAK)
         CALL EXTR_CP_GASBLKS_FROM_GENSYM_MAT(
     &        WORK(KLMAT1),WORK(KLMAT4),2,1,NGAS,1)
*. Diagonalize type-symmetry blocks of density matrix over active blocks
C             DIAG_BLK_SYMMAT(A,NBLK,LBLK,X,EIGENV,SCR,ISYM)
         CALL DIAG_BLK_SYMMAT(WORK(KLMAT4),NGAS*NGENSMOB,
     &        WORK(KLLEN_GASGS_AC_BLK),WORK(KLMAT2),WORK(KLMAT3),
     &        WORK(KLMAT5),1)
*. WORK(KLMAT2) contains eigenvector expansions
         IF(NTEST.GE.100) THEN
           WRITE(6,*)  ' Expansion of (pseudo-natural) orbitals'
           CALL APRBLM2(WORK(KLMAT2),WORK(KLLEN_GASGS_AC_BLK),
     &          WORK(KLLEN_GASGS_AC_BLK),NGAS*NGENSMOB,0)
         END IF
        END IF ! End if/else MO = 1, 5
*
*. pseudo-natural-canonical orbitals
*
        IF( IMO .EQ. 5 ) THEN
         STOP ' MOROT_GS: IMO = 5  not programmed '
        END IF ! (IMO=5)
*       ^ End if PS_NatCan orbital
      END IF
*     ^ end if some kind of natural orbitals were involved
*
* 3: Combine results from diagonalizing FIFA and RHO1 and save in MOMO
* ====================================================================
*
*
*. Add the act-act transformation to the total MOMO transformation matrix
      IF(IMO.EQ.1) THEN
*
*. Natural orbitals in active + pseudo canonical in inactive and sec
* ===================================================================
*
*
*. KLMAT2C contains canonical eigenvectors over TS-blocks
*. whereas KLMAT2 contains natural orbitals over Symmetry blocks
*. Expand KLMAT2C to full symmetry blocks
C       EXTR_CP_GASBLKS_FROM_GENSYM_MAT(AS,ASG,IEORC,IGAS_F,IGAS_L,IPAK)
        CALL EXTR_CP_GASBLKS_FROM_GENSYM_MAT(
     &       WORK(KLMAT4),WORK(KLMAT2C),1,0,NGAS+1,0)
*. And insert the natural orbitals
C ADD_ACTOB_TO_GAS_GNSYM_MAT(ATOT,AACT,IC_OR_E)
        CALL ADD_ACTOB_TO_GAS_GNSYM_MAT(WORK(KLMAT4),WORK(KLMAT2),1)
        CALL COPVEC(WORK(KLMAT4),WORK(KLMAT2C),LMOMO_GS)
      ELSE IF (IMO.EQ.3) THEN
*
*. KLMAT2C contains canonical eigenvectors over TS-blocks,
*. KLMAT2 contains natural orbitals over TS blocks
*. Insert active-active blocks from KLMAT2 into KLMAT2C
        CALL ADD_ACTOB_TO_GAS_GNSYM_MAT(WORK(KLMAT2C),WORK(KLMAT2),1)
*. Expand KLMAT2C to full symmetry blocks and save in KLMAT4
        ZERO = 0.0D0
        CALL SETVEC(WORK(KLMAT4),ZERO,LMOMO_GS)
C       EXTR_CP_GASBLKS_FROM_GENSYM_MAT(AS,ASG,IEORC,IGAS_F,IGAS_L,IPAK)
        CALL EXTR_CP_GASBLKS_FROM_GENSYM_MAT(
     &       WORK(KLMAT4),WORK(KLMAT2C),1,0,NGAS+1,0)
        CALL COPVEC(WORK(KLMAT4),WORK(KLMAT2C),LMOMO_GS)
      END IF
*
       IF(NTEST.GE.1000) THEN
         WRITE(6,*) ' Final transformation of MO-coefficients'
         CALL APRBLM2(WORK(KLMAT2C),NTOOB_GS,NTOOB_GS,NGENSMOB,0)
       END IF
*. And new MO-coefficients in general symmetry
      CALL MULT_BLOC_MAT(WORK(KLMAT2),WORK(KLMOMO_GN),WORK(KLMAT2C),
     &     NGENSMOB,NTOOB_GS,NTOOB_GS,NTOOB_GS,NTOOB_GS,NTOOB_GS,
     &     NTOOB_GS,0)
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Updated MO-MO matrix(general symmetry) '
        CALL APRBLM2(WORK(KLMAT2),NTOOB_GS,NTOOB_GS,NGENSMOB,0)
      END IF
*. And reform to standard symmetry, actual order
C     REFORM_MAT_STA_GEN(ASTA,AGEN,IPACK,IWAY)
      CALL REFORM_MAT_STA_GEN(WORK(KMOMO),WORK(KLMAT2),0,2)
*. Clean up
      CALL ISWPVE(IACT_TO_GENSM_REO, ISTA_TO_GENSM_REO, NTOOB)

      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' MO-MO transformation matrix MOMO(standard sym) '
        CALL APRBLM2(WORK(KMOMO),NTOOBS,NTOOBS,NSMOB,0)
      END IF
*
      IF(NOMOFL.EQ.0) THEN
*
*. Obtain input MO-AO transformation matrix in KMOAOUT
C?      CALL GET_CMOAO_ENV(WORK(KMOAOIN))
        CALL MULT_BLOC_MAT(WORK(KMOAOUT),WORK(KMOAOIN),WORK(KMOMO),
     &         NSMOB,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,NTOOBS,0)
*
        CALL COPVEC(WORK(KMOAOUT),WORK(KMOAO_ACT),LMOMO)
        IF(NTEST.GE.10) THEN
           WRITE(6,*) ' Output set of MO''s in required form'
           CALL PRINT_CMOAO(WORK(KMOAOUT))
        END IF
        IF(I_USE_SUPSYM.EQ.1.AND.IPRORB.GE.5) THEN
          WRITE(6,*) ' Output MO''s over shells '
C              PRINT_CMO_AS_SHELLS(CMO,IFORM)
          CALL PRINT_CMO_AS_SHELLS(WORK(KMOAO_ACT),2)
        END IF

*. Save on file LUMOUT
        CALL PUTMOAO(WORK(KMOAOUT))
      END IF
*     ^ End if MOAO file is present 
*
      CALL MEMMAN(IDUM,IDUM,'FLUSM ',IDUM,'MOROT ')
      RETURN
      END
      SUBROUTINE ADD_ACTOB_TO_MOMO_GNSYM(CMOMO_TOT,CMOMO_AC)
*
* A complete MOMO matrix CMOMO is given. Insert expansion of 
* active orbitals. General symmetry arrays used.
*
*. Jeppe Olsen, July 2012
*
*. Last revision: July 8, 2012
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cgas.inc'
*. Local scratch
      DIMENSION NTOOB_GS(MXP_NSUPSYM),NACOB_GS(MXP_NSUPSYM)
*. Input and output
      DIMENSION CMOMO_TOT(*)
*. Input
      DIMENSION CMOMO_AC(*)
*
*. Number of orbitals and active orbitals per standars symmetry
*
      NTEST = 100
      DO IGENSM = 1, NGENSMOB
        N = 0
        DO IGAS = 1, NGAS
          N = N + NGAS_GNSYM(IGENSM,IGAS)
        END DO
        NACOB_GS(IGENSM) = N
        NTOOB_GS(IGENSM) = N + NGAS_GNSYM(1,0) + NGAS_GNSYM(1,NGAS+1)
      END DO
*
      ZERO = 0.0D0
      IIOFF = 1
      IIOFF_AC = 1
      DO IGENSM = 1, NGENSMOB
        NOB_S = NTOOB_GS(IGENSM)
        NAC_S = NACOB_GS(IGENSM)
        NIN_S = NGAS_GNSYM(IGENSM,0)
        IAC = 0
        DO IORB = NIN_S+1,NIN_S + NAC_S
          IOFF = IIOFF + (IORB-1)*NOB_S
          IOFF_AC = IIOFF_AC + (IORB-NIN_S-1)*NAC_S
          CALL SETVEC(CMOMO_TOT(IOFF),ZERO,NOB_S)
          CALL COPVEC(CMOMO_AC(IOFF_AC),CMOMO_TOT(IOFF+NIN_S),NAC_S)
        END DO
        IIOFF = IIOFF + NOB_S**2
        IIOFF_AC = IIOFF + NAC_S**2
      END DO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Info from  ADD_ACTOB_TO_MOMO_GNSYM '
        WRITE(6,*)
        WRITE(6,*) ' CMOMO_AC matrix (Input) '
        CALL APRBLM2(CMOMO_AC,NACOB_GS,NACOB_GS,NGENSMOB,0)
        WRITE(6,*) ' CMOMO matrix (input and output) '
        CALL APRBLM2(CMOMO_TOT,NTOOB_GS,NTOOB_GS,NGENSMOB,0)
      END IF
*
      RETURN
      END
      SUBROUTINE ADD_ACTOB_TO_GAS_GNSYM_MAT(ATOT,AACT,IC_OR_E)
*
* A matrix ATOT containing diagonal types-general symmetry blocks 
* and a matrix AACT contaning these blocks for active orbitals 
* are given. Copy or extract the active blocks to/from ATOT from/to AACT
*
*. Jeppe Olsen, July 2012
*
*. Last modification: Sept. 24 2012, Jeppe Olsen, Debugged!
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'cgas.inc'
*. Input and output
      DIMENSION ATOT(*),AACT(*)
*
      NTEST = 00
*
      IIOFF = 1
      IIOFF_AC = 1
      DO IGENSM = 1, NGENSMOB
       DO IGAS = 0, NGAS + 1
        NOB = NGAS_GNSYM(IGENSM,IGAS)
        LBLK = NOB**2
        IF(0.LT.IGAS.AND.IGAS.LT.NGAS+1) THEN
          IF(IC_OR_E.EQ.1) THEN
            CALL COPVEC(AACT(IIOFF_AC),ATOT(IIOFF),LBLK)
          ELSE
            CALL COPVEC(ATOT(IIOFF),AACT(IIOFF_AC),LBLK)
          END IF
          IIOFF_AC = IIOFF_AC + LBLK 
        END IF
        IIOFF = IIOFF  + LBLK
       END DO
      END DO
*
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' ADD_ACTOB_TO_GAS_GNSYM_MAT speaking '
        IF(IC_OR_E.EQ.1) THEN
          WRITE(6,*) '  Active blocks copied from AACT to ATOT '
        ELSE
          WRITE(6,*) '  Active blocks extracted from ATOT to AACT '
        END IF
        WRITE(6,*) ' ATOT: '
C            WRT_SG_MAT(A,IS_OR_SG,IGAS_F,IGAS_L,IPAK,IEXT)
        CALL WRT_SG_MAT(ATOT,2,0,NGAS+1,0,1)
        WRITE(6,*) ' AACT: '
        CALL WRT_SG_MAT(AACT,2,1,NGAS,0,1)
      END IF
*
      RETURN
      END
      SUBROUTINE PNT4DMM(NO1PS,NO2PS,NO3PS,NO4PS,IDXSM,
     &           IS12,IS34,IS1234,IPNTR,ISM4A,NINT4D)
*
* Outer routine for determining pointers and dimension of integral 
* array
*
*. Jeppe Olsen, August 2012 ( about 16 years since the inner routine was written)
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'csmprd.inc'
      INCLUDE 'csm.inc'
*. Output
      INTEGER IPNTR(NSMOB,NSMOB,NSMOB)
      INTEGER ISM4A(NSMOB,NSMOB,NSMOB)
*
       CALL PNT4DM(NSMOB,NSMSX,MXPOBS,NO1PS,NO2PS,NO3PS,NO4PS,
     &      IDXSM,ADSXA,SXDXSX,IS12,IS34,IS1234,IPNTR,ISM4A,
     &      ADASX,NINT4D)
*
      RETURN
      END
      SUBROUTINE SELECT_ROOT(NROOT_ACT,ISROOT)
*
* NROOT_ACT CI vectors are Known. Select root to be used as reference root
*
*. Jeppe Olsen, Feb. 15. 2013.
*  Last modification; Jeppe Olsen;  Feb. 25, 2013
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'wrkspc-static.inc'
      INCLUDE 'crun.inc'
      INCLUDE 'cstate.inc'
      INCLUDE 'spinfo.inc'
      INCLUDE 'cicisp.inc'
      INCLUDE 'glbbas.inc'
*
      WRITE(6,*) ' ITG_SUPSYM in SELROT ', ITG_SUPSYM
*
*
*. 1: Select root:  two possibilities: Use root number from input or find first root with 
*.                  given supersymmetry
      IF(IROOT_MET(1:6).EQ.'SELORD') THEN
*. Use input root number 
        ISROOT = ITG_SROOT
      ELSE IF (IROOT_MET(1:6).EQ.'SELSPS') THEN
*. Find first root with supersymmetry INI_SUPSYM
*. Atomic or linear supersymmetry.
*. If supersymmetry has been specified for pointgroup, this is used.
*. Else linear supersymmetry is used
*. unless overall atomic supersymmetry has been specified - then this is used.
        IF(CSUPSYM(1:6).EQ.'ATOMIC') THEN
          KLLSUP = KL2EXP
        ELSE
          KLLSUP = KLZEXP
        END IF
        XVAL = FLOAT(ITG_SUPSYM)
        THRES = 0.4D0
C?      WRITE(6,*) ' First 6 elements of KLLSUP '
C?      CALL WRTMAT(WORK(KLLSUP),1,6,1,6)
*
C            FIND_XVAL_WITH_THRES(A,THRES, XVAL, NDIM,IVAL)
        CALL FIND_XVAL_WITH_THRES
     &       (dbl_mb(KLLSUP),THRES, XVAL,NROOT_ACT,IVAL)
        IF(IVAL.EQ.0) THEN
          WRITE(6,*) 
     &  ' No root with correct supersymmetry determined'
         WRITE(6,*) ' Supersymmetry, threshold ', XVAL,THRES
         WRITE(6,*) ' Supersymmetry of states: '
         CALL WRTMAT(dbl_mb(KLLSUP),1,NROOT_ACT,1,NROOT_ACT)
         STOP ' No root with correct supersymmetry determined'
        END IF
        ISROOT = IVAL
      END IF
*
      RETURN
      END
      SUBROUTINE PREPARE_NEW_LUC(LUCIN,LUCOUT,NROOTIN,NROOTUT,ISROOT,
     &           NVAR, ICISTR,ICOPY,VEC1)
*
* Prepare new file of CI vectors on LUCOUT from vectors in LUCIN so
* 1) Vector ISROOT on LUCIN as first vector
* 2) Lowest roots on LUCIN are the remaining vectors
*
*. Jeppe Olsen, Feb. 25, 2013
*
      INCLUDE 'implicit.inc'
*. Vector holding complete or block of vector
      DIMENSION VEC1(*)
*
      CALL REWINO(LUCIN)
      CALL REWINO(LUCOUT)
      LBLK = -1
*
      NTEST = 100 
      IF(NTEST.GE.100) THEN
        WRITE(6,*) ' Output from PREPARE_NEW_LUC '
        WRITE(6,*) ' ============================'
        WRITE(6,*) ' ISROOT = ', ISROOT 
        WRITE(6,*) ' NROOTUT, ICOPY = ', NROOTUT, ICOPY
        WRITE(6,*) ' ICISTR = ', ICISTR
      END IF
      IF(ICISTR.NE.1) THEN
*
*. Standard blocked storage
*
*. First vector ISROOT
        CALL SKPVCD(LUCIN,ISROOT-1,VEC1,1,LBLK) 
        WRITE(6,*) ' Vectors skipped '
        CALL COPVCD(LUCIN,LUCOUT,VEC1,0,LBLK)
*. And the remaning vectors
        CALL REWINO(LUCIN)
        IF(ISROOT.LE.NROOTUT) THEN
          LROOT = NROOTUT
        ELSE
          LROOT = NROOTUT-1
        END  IF
        DO IROOT = 1, LROOT
          IF(IROOT.NE.ISROOT) THEN
            CALL COPVCD(LUCIN,LUCOUT,VEC1,0,LBLK)
          ELSE
            CALL SKPVCD(LUCIN,1,VEC1,0,LBLK) 
          END IF
        END DO
*
        IF(ICOPY.EQ.1 ) THEN
*. Copy back
         CALL REWINO(LUCIN)
         CALL REWINO(LUCOUT)
         DO IROOT = 1, NROOTUT
            CALL COPVCD(LUCOUT,LUCIN,VEC1,0,LBLK)
         END DO
        END IF ! I should copy back
      ELSE
*. Vectors are stored in one block
        DO IVEC = 1, ISROOT
            CALL FRMDSC(VEC1,NVAR,-1  ,LUCIN,IMZERO,IAMPACK)
        END DO
        CALL TODSC(VEC1,NVAR,-1  ,LUCOUT)
        IF(ISROOT.LE.NROOTUT) THEN
          LROOT = NROOTUT
        ELSE
          LROOT = NROOTUT-1
        END IF
        CALL REWINO(LUCIN)
        DO IROOT = 1, LROOT
          CALL FRMDSC(VEC1,NVAR,-1  ,LUCIN,IMZERO,IAMPACK)
          IF(IROOT.NE.ISROOT) THEN
            CALL TODSC(VEC1,NVAR,-1  ,LUCOUT)
          END IF
        END DO
*
        IF(ICOPY.EQ.1) THEN
*. Copy back
         CALL REWINO(LUCIN)
         CALL REWINO(LUCOUT)
         DO IROOT = 1, NROOTUT
            CALL FRMDSC(VEC1,NVAR,-1  ,LUCOUT,IMZERO,IAMPACK)
            CALL TODSC(VEC1,NVAR,-1  ,LUCIN)
         END DO
        END IF ! I should copy back
      END IF ! ICISTR switch
*
      RETURN
      END 
      SUBROUTINE PRINT_CMOAO_FRAG(CMOAO,IFRAG)
*
* Print MO-AO expansion coefficient matrix CMOAO for fragment IFRAG
*
*. Jeppe Olsen, March 2013, from PRINT_CMOAO
*
      INCLUDE 'implicit.inc'
      INCLUDE 'mxpdim.inc'
      INCLUDE 'lucinp.inc'
      INCLUDE 'orbinp.inc'
      INCLUDE 'fragmol.inc'
*. Specific input
      DIMENSION CMOAO(*)
*. General input
      CHARACTER*4 AO_CENT, AO_TYPE
      COMMON/AOLABELS/AO_CENT(MXPORB),AO_TYPE(MXPORB)
*
      WRITE(6,*) ' MO - AO transformation matrix  for fragment ', IFRAG
      WRITE(6,*) ' ----------------------------------------------------'
      NMO_PER_BLK = 10 
      IOFF_AO = 1
      IOFF_MOAO = 1
      DO ISM = 1, NSYM_FRAG(IFRAG)
        WRITE(6,*)
        WRITE(6,*) ' MO''s with symmetry ', ISM
        WRITE(6,*)
        NMOS = NBAS_FRAG(ISM,IFRAG) 
        NAOS = NBAS_FRAG(ISM,IFRAG)
        NBLOCK = NMOS/NMO_PER_BLK
        IF(NBLOCK*NMO_PER_BLK.LT.NMOS) NBLOCK = NBLOCK + 1
C?      WRITE(6,*) ' ISM, NMOS, NAOS, NBLOCK = ',
C?   &               ISM, NMOS, NAOS, NBLOCK
        DO IBLOCK = 1, NBLOCK
          IMO_START = (IBLOCK-1)*NMO_PER_BLK  + 1
          IMO_STOP  = MIN(NMOS,IBLOCK*NMO_PER_BLK)
          WRITE(6,'(14X, 10(2X,I4,1X))') (IMO, IMO = IMO_START,IMO_STOP)
          DO IAO = 1, NAOS
           WRITE(6,'(2X,A4,2X,A4,2X,10F7.3,(14X,10F7.3))')
     &     AO_CENT_FRAG(IOFF_AO-1+IAO,IFRAG), 
     &     AO_TYPE_FRAG(IOFF_AO-1+IAO,IFRAG),
     &     (CMOAO(IOFF_MOAO-1+(IMO-1)*NAOS+IAO),IMO =IMO_START,IMO_STOP)
          END DO
        END DO
        IOFF_AO = IOFF_AO + NAOS
        IOFF_MOAO = IOFF_MOAO + NMOS*NAOS
      END DO
*
      RETURN
      END

      SUBROUTINE ORDSTR_GEN(IINST,NELMNT,IORD,ISIGN,IOUTST,
     &           IPLACE,IPRNT)
*
* Order a string of integers in ascending order of a general
* order array 
*
*. Input:
C
C IINST: INPUT STRING IS IINST
C IOUTST: OUTPUT STRING IS IOUTST
C NELMNT: NUMBER OF INTEGERS IN STRING
C ISIGN:  SIGN OF PERMUTATION: + 1: EVEN PERMUTATIONN
C                                - 1: ODD  PERMUTATION
* IORD: The ordering array giving the required order of the integers
*       - Yes, we are getting general and advanced
*
*. Output:
*
* Reordered array is given in IOUTST
* IPLACE :original position of each new position 
*
*. This code is still bases on the order code of Joe Golab..
*
*. Jeppe Olsen, June 2, 2013; Sitting in Lugano, preparing for a talk
*
      IMPLICIT REAL*8           (A-H,O-Z)
      DIMENSION IINST(NELMNT),IOUTST(NELMNT), IORD(*)
      DIMENSION IPLACE(NELMNT)
C
      NTEST = 00
      IF(NTEST.GE.100) THEN
       WRITE(6,*) ' Output from  ORDSTR_GEN: '
       WRITE(6,*) ' Number of elements to be sorted ', NELMNT
       WRITE(6,*) ' And the elements ' 
       CALL IWRTMA(IINST,1,NELMNT,1,NELMNT)
      END IF
*
      ISIGN = 1
      DO I = 1, NELMNT
        IPLACE(I) = I
      END DO
      IF(NELMNT.EQ.0) GOTO 50
*
      CALL ICOPVE(IINST,IOUTST,NELMNT)
      ISIGN = 1
C
C       BEGIN TO ORDER
C
        JOE = 1
  10    I = JOE
  20    CONTINUE
        IF(I.EQ.NELMNT) GO TO 50
        IF(IORD(IOUTST(I)).LE.IORD(IOUTST(I+1))) GO TO 40
        JOE = I + 1
  30    SWAP = IOUTST(I)
        ISIGN = - ISIGN
        IOUTST(I) = IOUTST(I+1)
        IOUTST(I+1) = SWAP
        SWAP = IPLACE(I)
        IPLACE(I) = IPLACE(I+1)
        IPLACE(I+1) = SWAP
        IF(I.EQ.1) GO TO 10
        I = I - 1
        IF(IORD(IOUTST(I)).GT.IORD(IOUTST(I+1))) GO TO 30
        GO TO 10
 40     I = I + 1
      GO TO 20
C
C     END ORDER
C
 50   CONTINUE
      NTEST = MAX(NTEST,IPRNT)
      IF( NTEST .GE.200) THEN
        WRITE(6,*)  ' Input string, ordered string: '
        CALL IWRTMA(IINST,1,NELMNT,1,NELMNT)
        CALL IWRTMA(IOUTST,1,NELMNT,1,NELMNT)
        WRITE(6,*) ' ISIGN: ', ISIGN
        WRITE(6,*) ' Original position of each reordered position'
        CALL IWRTMA(IPLACE,1,NELMNT,1,NELMNT)
      END IF
C
      RETURN
C
      END
