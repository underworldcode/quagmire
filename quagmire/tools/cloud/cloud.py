# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
# 
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.


"""
We use the python filesystem `fs` package to provide simple and similar access patterns for files stored in the cloud.

By default, we return a handle to access a cloudstor directory where read-only examples, 
meshes, pre-computed topography etc can be provided. These are typically too large to include in github. 
"""

# Let the fs package be directly available to users

import fs


def verify_cloud_fs(cloud_fs_location):
    """Verify that cloud_fs_location is a valid filesystem type"""

    if isinstance(cloud_fs_location, fs.base.FS):
        cloud_fs_location.check()
        return True

    # this may have raised an error already ?
    return False 

def cloud_download(cloud_filename, local_filename, cloud_fs_location_handle=None):
        """
        Load a file from a cloud_location pointed to by
        a pyfilesystem object. (By default, the quagmire cloud service on cloudstor)

        Parameters
        ----------

        cloud_fs_location_handle: fs.base.FS


        cloud_filename: str
            The filename for the file in the cloud. It should be possible to do 
            a cloud_location_handle.listdir(".") to check the file names

        local_filename: str
            The local destination for the cloud file. No error checking is performed - see
            FS.download() for more information on usage / errors
        """

        if cloud_fs_location_handle is None:
           cloud_fs_location_handle = quagmire_cloud_fs

        try:
            verify_cloud_fs(cloud_fs_location_handle)
        except:
            print("Not a valid cloud location - no files available", flush=True)
            return

        if not cloud_fs_location_handle.exists(str(cloud_filename)):
            print("Not a valid cloud file - {}".format(str(cloud_filename)), flush=True)
            return



        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            with open(str(local_filename), 'wb') as local_file:
                cloud_fs_location_handle.download(str(cloud_filename), local_file)

        comm.barrier()

        return

def google_drive_convert_link(share_url):
    """ Given a url that is returned from a google drive "share this file" link, convert to 
    a downloadable url that works with  `cloud_download`.

    Note that google drive files are limited in size to 100Mb for direct download. 
    Permissions need to be set to 'anyone with this link can view' otherwise the download will
    fail (no authorization) 
    """

    u1 = share_url.split('/d/')[1]
    u2 = u1.split("/")[0]
    
    download_url = "https://drive.google.com/uc?export=download&id={}".format(u2)
    
    return download_url
    


def url_download(url, local_filename):
    """Given a publicly accessible url to a file, download the file to the local_filename"""

    import requests
    filepath=local_filename

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:

        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get(url, stream=True, headers=headers)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)

    comm.barrier()

    return


def cloud_upload(cloud_filename, local_filename, cloud_fs_location_handle):
        """
        Load a file from a cloud_location pointed to by
        a pyfilesystem object. (By default, the quagmire cloud service on cloudstor)

        Parameters
        ----------

        cloud_fs_location_handle: fs.base.FS

        cloud_filename: str
            The filename for the file in the cloud. It should be possible to do 
            a cloud_location_handle.listdir(".") to check the file names

        local_filename: str
            The local destination for the cloud file. No error checking is performed - see
            FS.download() for more information on usage / errors
        """

        try:
            verify_cloud_fs(cloud_fs_location_handle)
        except:
            print("Not a valid cloud location - no data was uploaded", flush=True)
            return

        # Note - this is not reliable and is unlikely to fail until the verification step
        if cloud_fs_location_handle.getmeta()["read_only"]:
            print("The cloud location you have provided is read only - no data was uploaded", flush=True)
            return

        if cloud_fs_location_handle == quagmire_cloud_fs:
            print("The quagmire cloud is read only - no data was uploaded", flush=True)
            return

    
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            with open(local_filename, 'rb') as local_file:
                cloud_fs_location_handle.upload(cloud_filename, local_file)

        comm.barrier()

        # Verify upload 
        if not  cloud_fs_location_handle.exists(cloud_filename):
            print("Warning: Upload can't be verified for file {}".format(cloud_filename), flush=True)



        return


def cloudstor(private=False, url=None, password=None, username=None):
        """Return a cloudstor (webdav) pyfilesystem handle for public or private
           access. Password is not stored anywhere unless you provide it in your
           script. It is not advisable to code a password directly in any script
           that you publish anywhere.

           Cloudstore allows public access to any file or directory with read-only or
           read-write options and (temporary) password protection if needed.

           You can also obtain an app password that you can control separately to
           your online account password but this still provides unlimited access to
           your files.

           You can also provide any password information via environment variables
           if you prefer (see Examples)
        """

        import fs
        # from requests.auth import HTTPBasicAuth
        import getpass

        if not private:
            if url is None:
                print("The public url to the cloudstor shared resource is required")
                print("It should be of the form:")
                print("\t https://cloudstor.aarnet.edu.au/plus/s/UNoxqDf8nevZk")
                print("\t or just specify the key string ....... UNoxqDf8nevZk")
                print("\n")
                raise(RuntimeError("The public url to the cloudstor shared resource is required"))
            else:
                username = url.replace("https://cloudstor.aarnet.edu.au/plus/s/","")

            public_url = "https://cloudstor.aarnet.edu.au/plus/s/" + username

            if password is None:
                password = getpass.getpass("Password for {}".format(public_url))

            webdav_url = "webdav://{}:{}@cloudstor.aarnet.edu.au:443/plus/public.php/webdav/".format(username, password)

        else:
            if username is None:
                raise(RuntimeError("The username for the cloudstore resource is required"))

            url = "https://cloudstor.aarnet.edu.au"

            if password is None:
                password = getpass.getpass("Cloudstore password for {}".format(username))

            webdav_url = "webdav://{}:{}@cloudstor.aarnet.edu.au:443/plus/remote.php/webdav/".format(username, password)

        return fs.open_fs(webdav_url)


# _underworldcode_cloudstor_fs = cloudstor(url="https://cloudstor.aarnet.edu.au/plus/s/ladjou9ky26ZG0A", password="8M7idzp2Q7DXLMz()()()()()")
_underworldcode_cloudstor_fs = cloudstor(url="https://cloudstor.aarnet.edu.au/plus/s/4SEAhkqSlTojYhv", password="8M7idzp2Q7DXLMz()()()()()")
quagmire_cloud_fs = _underworldcode_cloudstor_fs.opendir('/')

# This is a file drop location where developers can read 
_underworldcode_cloudstor_fs = cloudstor(url="https://cloudstor.aarnet.edu.au/plus/s/eyOWXBY1u4bIrLg", password="8M7idzp2Q7DXLMz()()()()()")
quagmire_cloud_filedrop_fs = _underworldcode_cloudstor_fs.opendir('/')

del _underworldcode_cloudstor_fs

# This is a temporary directory that we can pass around 
# and use for unpacking cloud files etc. In parallel, 
# we need a filesystem directory and name to pass around

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    _quagmire_cloud_cache_dir = fs.open_fs("temp://_quagmire_cache_dir")
    _quagmire_cloud_cache_root_path = _quagmire_cloud_cache_dir.root_path
else:
    _quagmire_cloud_cache_root_path = None

comm.barrier()


quagmire_cloud_cache_dir_name = comm.bcast(_quagmire_cloud_cache_root_path, root=0)



