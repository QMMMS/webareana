# gdown <file_id>

docker start shopping
docker start shopping_admin
docker start forum
docker start gitlab
docker start wikipedia

# OneStopShop

curl -C - -H "Authorization: Bearer ya29.a0AQQ_BDRBKsZIdVSX9zJ3hWQIT9wpzea1u70178-Lwama9x1FRGLi_GS5BKAtXIHUOVhqL-y8sNLa9d8-RAhs816-Jksl-qL9OUrYYaARn5Ty6qM4WRFLD5si3vMY2lYPQIJVf9SRG78-0WfSjev5qtgUyQrpB7bER48-jZmsQ_UGWuOccf0P_zOkB54Vswquyqn_b70aCgYKAUgSARQSFQHGX2MiEM44NWV5hZOxscLbdDqHYA0206" https://www.googleapis.com/drive/v3/files/1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA?alt=media -o shopping_final_0712.tar
docker load --input shopping_final_0712.tar
docker run --name shopping -p 7770:80 -d shopping_final_0712
# wait ~1 min to wait all services to start

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://10.130.138.30:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://10.130.138.30:7770/" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush


# CMS
docker load --input shopping_admin_final_0719.tar
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
# wait ~1 min to wait all services to start

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://10.130.138.30:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://10.130.138.30:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush


# Reddit
curl -C - -H "Authorization: Bearer ya29.a0AQQ_BDRBKsZIdVSX9zJ3hWQIT9wpzea1u70178-Lwama9x1FRGLi_GS5BKAtXIHUOVhqL-y8sNLa9d8-RAhs816-Jksl-qL9OUrYYaARn5Ty6qM4WRFLD5si3vMY2lYPQIJVf9SRG78-0WfSjev5qtgUyQrpB7bER48-jZmsQ_UGWuOccf0P_zOkB54Vswquyqn_b70aCgYKAUgSARQSFQHGX2MiEM44NWV5hZOxscLbdDqHYA0206" https://www.googleapis.com/drive/v3/files/17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf?alt=media -o postmill-populated-exposed-withimg.tar

docker load --input postmill-populated-exposed-withimg.tar
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg

###########如果以及启动过，可以#############
docker ps -a
docker start <container_name>
##########如果要重置环境，可以#############
docker stop <container_name>
docker rm <container_name>
# 按照上面 run 的命令重新启动



# gitlab 
# curl -C - -O http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar
docker load --input gitlab-populated-final-port8023.tar
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start

# wait at least 5 mins for services to boot
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://10.130.138.30:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure


# Wikipedia
curl -C - -H "Authorization: Bearer ya29.a0AQQ_BDR00GrRz-aKdM0D-rO0Ju9p15zl-2um-jIccfhHPsL9CNzSJJNAeFo7o05zM-OpAotjuqAC9uWbmkIUDoxyKhEuTJLO3p87gyUok8rYVilOgV9HV5sKQQ6GKhVEhaKg8p-eAfGrxGTWY4AIeUYJixItfBER7LU6fApvEpKKcFYeCnMV9EqVLvVSHMYBlPaUvmUaCgYKAQwSARQSFQHGX2Mi5fd-D8yzyM-DJ4CXgbBcwg0206" https://www.googleapis.com/drive/v3/files/1Um4QLxi_bGv5bP6kt83Ke0lNjuV9Tm0P?alt=media -o wikipedia_en_all_maxi_2022-05.zim
docker run -d --name=wikipedia --volume=/home/zjusst/qms/docker_envs/:/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim



# map

sudo docker compose build


sudo sed -i 's|http://ogma.lti.cs.cmu.edu:8080|http://18.208.187.221:8080|g' /home/zjusst/qms/docker_envs/openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js
sudo sed -i 's|metis.lti.cs.cmu.edu:8085|18.208.187.221:8085|g' /home/zjusst/qms/docker_envs/openstreetmap-website/config/settings.yml
sudo sed -i 's|metis.lti.cs.cmu.edu:|18.208.187.221:|g' /home/zjusst/qms/docker_envs/openstreetmap-website/config/settings.yml
cd /home/zjusst/qms/docker_envs/openstreetmap-website/ && docker compose restart web




# homepage
YOUR_ACTUAL_HOSTNAME=http://10.130.138.30
YOUR_ACTUAL_HOSTNAME=${YOUR_ACTUAL_HOSTNAME%/}
perl -pi -e "s|<your-server-hostname>|${YOUR_ACTUAL_HOSTNAME}|g" webarena-homepage/templates/index.html
cd webarena-homepage
flask run --host=0.0.0.0 --port=4399

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}