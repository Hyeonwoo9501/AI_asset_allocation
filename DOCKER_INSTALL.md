# Docker 설치 가이드

이 문서는 Windows WSL2 환경에서 Docker를 설치하는 방법을 설명합니다.

## 방법 1: Docker Desktop for Windows (권장)

Docker Desktop은 Windows와 WSL2를 자동으로 통합해주는 가장 쉬운 방법입니다.

### 설치 단계

1. **Docker Desktop 다운로드**
   - https://www.docker.com/products/docker-desktop/ 방문
   - "Download for Windows" 클릭

2. **설치 실행**
   - 다운로드한 `Docker Desktop Installer.exe` 실행
   - 설치 중 "Use WSL 2 instead of Hyper-V" 옵션 선택
   - 설치 완료 후 시스템 재시작

3. **Docker Desktop 설정**
   - Docker Desktop 실행
   - Settings > Resources > WSL Integration으로 이동
   - "Enable integration with my default WSL distro" 활성화
   - Ubuntu 배포판 선택 및 활성화
   - "Apply & Restart" 클릭

4. **설치 확인**
   ```bash
   # WSL2 터미널에서 실행
   docker --version
   docker-compose --version
   ```

   성공적으로 설치되면 버전 정보가 출력됩니다.

## 방법 2: WSL2에 직접 Docker Engine 설치

Docker Desktop 없이 WSL2에 직접 설치하는 방법입니다.

### 설치 단계

1. **이전 버전 제거**
   ```bash
   sudo apt-get remove docker docker-engine docker.io containerd runc
   ```

2. **필수 패키지 설치**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       ca-certificates \
       curl \
       gnupg \
       lsb-release
   ```

3. **Docker GPG 키 추가**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

4. **Docker 저장소 추가**
   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. **Docker Engine 설치**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

6. **Docker 서비스 시작**
   ```bash
   sudo service docker start
   ```

7. **사용자를 docker 그룹에 추가 (sudo 없이 사용)**
   ```bash
   sudo usermod -aG docker $USER
   ```

   이후 WSL2 재시작:
   ```bash
   # Windows PowerShell에서 실행
   wsl --shutdown
   # 그 다음 WSL2 다시 시작
   ```

8. **설치 확인**
   ```bash
   docker --version
   docker compose version
   docker run hello-world
   ```

### WSL2에서 Docker 자동 시작 설정

WSL2는 기본적으로 systemd를 사용하지 않으므로, Docker를 수동으로 시작해야 합니다.

방법 1: 매번 수동 시작
```bash
sudo service docker start
```

방법 2: ~/.bashrc에 추가하여 자동 시작
```bash
echo 'sudo service docker start' >> ~/.bashrc
```

방법 3: WSL2에서 systemd 활성화 (Ubuntu 22.04+)
```bash
# /etc/wsl.conf 파일 생성/수정
sudo nano /etc/wsl.conf

# 다음 내용 추가:
[boot]
systemd=true

# WSL2 재시작 (Windows PowerShell)
wsl --shutdown
```

## 설치 확인 및 테스트

1. **Docker 버전 확인**
   ```bash
   docker --version
   docker compose version
   ```

2. **Hello World 실행**
   ```bash
   docker run hello-world
   ```

3. **Docker 정보 확인**
   ```bash
   docker info
   ```

## 일반적인 문제 해결

### "Cannot connect to the Docker daemon" 오류

```bash
# Docker 서비스가 실행 중인지 확인
sudo service docker status

# 실행 중이 아니면 시작
sudo service docker start
```

### "permission denied" 오류

```bash
# docker 그룹에 사용자 추가
sudo usermod -aG docker $USER

# WSL2 재시작
wsl --shutdown  # Windows PowerShell에서
```

### WSL2 메모리 제한

Docker가 너무 많은 메모리를 사용하는 경우, Windows에서 `.wslconfig` 파일 생성:

```
# C:\Users\YourUsername\.wslconfig
[wsl2]
memory=8GB
processors=4
```

## 다음 단계

Docker 설치가 완료되면 프로젝트를 실행할 수 있습니다:

```bash
cd /mnt/c/projects/AI_asset_allocation

# 이미지 빌드
./docker_run.sh build

# 컨테이너 실행
./docker_run.sh cpu

# 또는 직접 실행
docker-compose up -d transformer-cpu
docker-compose exec transformer-cpu bash
```

## 참고 자료

- Docker Desktop: https://docs.docker.com/desktop/windows/wsl/
- Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- WSL2 Documentation: https://docs.microsoft.com/en-us/windows/wsl/
