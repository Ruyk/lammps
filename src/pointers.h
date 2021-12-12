/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
// clang-format off

// Pointers class contains ptrs to master copy of
//   fundamental LAMMPS class ptrs stored in lammps.h
// every LAMMPS class inherits from Pointers to access lammps.h ptrs
// these variables are auto-initialized by Pointer class constructor
// *& variables are really pointers to the pointers in lammps.h
// & enables them to be accessed directly in any class, e.g. atom->x

#ifndef LMP_POINTERS_H
#define LMP_POINTERS_H

#include "lmptype.h"    // IWYU pragma: export
#include <mpi.h>        // IWYU pragma: export
#include <cstddef>      // IWYU pragme: export
#include <cstdio>       // IWYU pragma: export
#include <string>       // IWYU pragma: export
#include "lammps.h"     // IWYU pragma: export
#include "utils.h"      // IWYU pragma: export
#include "fmt/format.h" // IWYU pragma: export

namespace LAMMPS_NS {

// universal defines inside namespace

#define FLERR __FILE__,__LINE__

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

// enum used for KOKKOS host/device flags

enum ExecutionSpace{ Host, Device };

// global forward declarations

template <class T> class MyPoolChunk;
template <class T> class MyPage;

/** \class LAMMPS_NS::Pointers
 * \brief Base class for LAMMPS features
 *
 * The Pointers class contains references to many of the pointers
 * and members of the LAMMPS_NS::LAMMPS class. Derived classes thus
 * gain access to the constituent class instances in the LAMMPS
 * composite class and thus to the core functionality of LAMMPS.
 *
 * This kind of construct is needed, since the LAMMPS constructor
 * should only be run once per LAMMPS instance and thus classes
 * cannot be derived from LAMMPS itself. The Pointers class
 * constructor instead only initializes C++ references to component
 * pointer in the LAMMPS class. */

// get rid of all the references so we can compile this with offload
// compiler. The device better not actually use any of them.

class Pointers;

template <typename Return, typename Containing, Return &(Containing::* get)()>
struct proxy
{
   Containing *cp;
   proxy(Containing& c) : cp(&c) {}
   operator Return() { return ((*cp).*get)(); }
   Return &operator=(Return r) { return ((*cp).*get)() = r; }
   Return operator->() const { return ((*cp).*get)(); }
};

struct atomproxy
{
    typedef Atom *AtomPtr;
    AtomPtr *atomptr;
    atomproxy(AtomPtr &a) : atomptr(&a) {};
    operator AtomPtr() { return (*atomptr); }
    AtomPtr operator->() const { return (*atomptr); }
    operator AtomKokkos *() const { return (AtomKokkos *)(*atomptr); }
    AtomPtr& operator=(AtomKokkos *a) { return (*atomptr) = (Atom *)a; }
};
struct memoryproxy
{
    typedef Memory *MemoryPtr;
    MemoryPtr *memoryptr;
    memoryproxy(MemoryPtr &a) : memoryptr(&a) {};
    operator MemoryPtr() { return (*memoryptr); }
    MemoryPtr operator->() const { return (*memoryptr); }
    operator MemoryKokkos *() const { return (MemoryKokkos *)(*memoryptr); }
};

class NeighborKokkos;
struct neighborproxy
{
    typedef Neighbor *NeighborPtr;
    NeighborPtr *neighborptr;
    neighborproxy(NeighborPtr &a) : neighborptr(&a) {};
    operator NeighborPtr() { return (*neighborptr); }
    NeighborPtr operator->() const { return (*neighborptr); }
    operator NeighborKokkos *() const
        { return (NeighborKokkos *)(*neighborptr); }
};

class DomainKokkos;
struct domainproxy
{
    typedef Domain *DomainPtr;
    DomainPtr *domainptr;
    domainproxy(DomainPtr &a) : domainptr(&a) {};
    operator DomainPtr() { return (*domainptr); }
    DomainPtr operator->() const { return (*domainptr); }
    operator DomainKokkos *() const
        { return (DomainKokkos *)(*domainptr); }
};

class CommKokkos;
class CommBrick;
class CommTiled;
struct commproxy
{
    typedef Comm *CommPtr;
    CommPtr *commptr;
    commproxy(CommPtr &a) : commptr(&a) {};
    operator CommPtr() { return (*commptr); }
    CommPtr operator->() const { return (*commptr); }
    operator CommBrick *() const
        { return (CommBrick *)(*commptr); }
    operator CommTiled *() const
        { return (CommTiled *)(*commptr); }
    operator CommKokkos *() const
        { return (CommKokkos *)(*commptr); }
};


class Pointers {
  Error *&get_error() { return lmp->error; }
  Universe *&get_universe() { return lmp->universe; }
  Input *&get_input() { return lmp->input; }

  Update *&get_update() { return lmp->update; }
  Force *&get_force() { return lmp->force; }
  Modify *&get_modify() { return lmp->modify; }
  Group *&get_group() { return lmp->group; }
  Output *&get_output() { return lmp->output; }
  Timer *&get_timer() { return lmp->timer; }

  MPI_Comm &get_world() { return lmp->world; }
  FILE *&get_infile() { return lmp->infile; }
  FILE *&get_screen() { return lmp->screen; }
  FILE *&get_logfile() { return lmp->logfile; }

  class AtomKokkos *&get_atomKK() { return lmp->atomKK; }
  class MemoryKokkos *&get_memoryKK() { return lmp->memoryKK; }
  class Python *&get_python() { return lmp->python; }

 public:
  Pointers(LAMMPS *ptr) : lmp(ptr),
    memory(ptr->memory),
    error(*this),
    universe(*this),
    input(*this),

    atom(ptr->atom),
    update(*this),
    neighbor(ptr->neighbor),
    comm(ptr->comm),
    domain(ptr->domain),
    force(*this),
    modify(*this),
    group(*this),
    output(*this),
    timer(*this),

    world(*this),
    infile(*this),
    screen(*this),
    logfile(*this),

    atomKK(*this),
    memoryKK(*this),
    python(*this) {}

  virtual ~Pointers() {}

 protected:
  LAMMPS *lmp;
  memoryproxy memory;
  proxy<Error *, Pointers, &Pointers::get_error>  error;
  proxy<Universe *, Pointers, &Pointers::get_universe> universe;
  proxy<Input *, Pointers, &Pointers::get_input> input;

  atomproxy atom;
  proxy<Update *, Pointers, &Pointers::get_update> update;
  neighborproxy neighbor;
  commproxy comm;
  //proxy<Comm *, Pointers, &Pointers::get_comm> comm;
  //proxy<Domain *, Pointers, &Pointers::get_domain> domain;
  domainproxy domain;
  proxy<Force *, Pointers, &Pointers::get_force> force;
  proxy<Modify *, Pointers, &Pointers::get_modify> modify;
  proxy<Group *, Pointers, &Pointers::get_group> group;
  proxy<Output *, Pointers, &Pointers::get_output> output;
  proxy<Timer *, Pointers, &Pointers::get_timer> timer;

  proxy<MPI_Comm, Pointers, &Pointers::get_world> world;
  proxy<FILE *, Pointers, &Pointers::get_infile> infile;
  proxy<FILE *, Pointers, &Pointers::get_screen> screen;
  proxy<FILE *, Pointers, &Pointers::get_logfile> logfile;

  proxy<class AtomKokkos *, Pointers, &Pointers::get_atomKK> atomKK;
  proxy<class MemoryKokkos *, Pointers, &Pointers::get_memoryKK> memoryKK;
  proxy<class Python *, Pointers, &Pointers::get_python> python;
};

}

#endif
