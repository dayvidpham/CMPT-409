{
  description = "Provides a reproducible python3 env to run the simulations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:NixOS/nixpkgs/a2eb207f45e4a14a1e3019d9e3863d1e208e2295";

    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    inputs@{ self
    , nixpkgs
    , nixpkgs-unstable
    , nixpkgs-python
    , flake-utils
    , ...
    }:
    let
      mkEnvFromChannel = (nixpkgs-channel: nixpkgs-channel-python:
        flake-utils.lib.eachDefaultSystem (system:
          let
            pkgs = import nixpkgs-channel {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = true;
              config.cudaVersion = "12";
            };
            pkgs-python = import nixpkgs-channel-python {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = true;
              config.cudaVersion = "12";
            };

            #########################################
            # Defines python dependencies

            python3-pkgName = "python310";

            f-python3-prodPkgs = (python-pkgs: (with python-pkgs; [
              matplotlib
              numpy
              torch
              pandas
              scikit-learn
            ]) ++ [
            ]);

            f-python3-buildInputs = (pkgs_: with pkgs_; [
              zlib
              glibc
              stdenv.cc.cc.lib
              gcc
              tk
              tcl
              pkgs_.libxcrypt
            ]);


            #nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;
            f-nvidia-buildInputs = (pkgs_: with pkgs_; [
              ffmpeg
              fmt.dev
              #cudaPackages.cuda_cudart
              #cudatoolkit
              #nvidiaPackage
              #cudaPackages.cudnn
              libGLU
              libGL
              xorg.libXi
              xorg.libXmu
              freeglut
              xorg.libXext
              xorg.libX11
              xorg.libXv
              xorg.libXrandr
              zlib
              ncurses
              stdenv.cc
              binutils
              uv
              wayland
            ]);

            f-nvidia-shellHook = (pkgs_: with pkgs_; ''
              export CMAKE_PREFIX_PATH="${pkgs_.fmt.dev}:$CMAKE_PREFIX_PATH"
              export PKG_CONFIG_PATH="${pkgs_.fmt.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
            '');


            #########################################
            # Will be used to define
            # outputs.devShells

            devShell = pkgs.mkShell {
              name = "cmpt409-project";

              buildInputs = with pkgs; [
                ffmpeg
                fmt.dev
                libGLU
                libGL
                glew
                glfw3
                mesa
                xorg.libXi
                xorg.libXmu
                freeglut
                xorg.libXext
                xorg.libX11
                xorg.libXv
                xorg.libXrandr
                zlib
                ncurses
                stdenv.cc
                binutils
                wayland

                uv
                zlib
                glibc
                stdenv.cc.cc.lib
                gcc
                tk
                tcl
                libxcrypt
                libxcrypt-legacy
              ];

              packages = [
                pkgs.micromamba
                pkgs.uv
                pkgs-python.python310
              ];

              shellHook =
                let
                  tk = pkgs.tk;
                  tcl = pkgs.tcl;
                in
                ''
                  export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH"
                  export TK_LIBRARY="${tk}/lib/${tk.libPrefix}"
                  export TCL_LIBRARY="${tcl}/lib/${tcl.libPrefix}"

                  ${f-nvidia-shellHook pkgs}
                '';

              allowSubstitutes = false;
            };

            fhs = (pkgs.buildFHSEnv
              {
                name = "cmpt409-project-fhs";
                targetPkgs = (fhs-pkgs:
                  [
                    fhs-pkgs.uv
                    fhs-pkgs.git
                  ]
                  ++ (f-python3-buildInputs fhs-pkgs)
                  ++ (f-nvidia-buildInputs fhs-pkgs)
                );

                multiPkgs = fhs-pkgs: with fhs-pkgs; [
                  zlib
                  libxcrypt-legacy
                ];

                runScript = "bash -i";

                profile = ''
                  ${f-nvidia-shellHook pkgs}
                '';

                allowSubstitutes = false;
              }).env;
          in
          {
            devShells.default = devShell;
            devShells.fhs = fhs;
          }
        ));
    in
    mkEnvFromChannel
      nixpkgs-unstable
      nixpkgs-unstable;
}
